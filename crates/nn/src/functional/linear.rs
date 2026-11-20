//! Linear projections `out = x · weightᵀ + bias`, one function per representation a checkpoint
//! stores a weight in.
//!
//! ```text
//! linear          a dense weight: one `gemm`
//! linear_fp8      a per-tensor FP8 weight: the activation encoded (`fp8_quantize`) and the
//!                 product over the two encodings (`fp8_gemm`)
//! linear_nvfp4    an NVFP4 weight: the activation encoded (`nvfp4_quantize`, the encoding of
//!                 [`super::nvfp4_quantize`]) and the packed product (`nvfp4_gemm`)
//! dynamic_scales  the two factors an encoded product reads, measured from its activation:
//!                 each row's largest magnitude (`reduce`'s `absmax`) folded with the weight's
//!                 (`dynamic_scale`)
//! ```
//!
//! The encoded products read their activation's scale as two `[1]` F32 device words: the
//! encoder's multiplier and the product's factor. A calibrated export publishes both at load;
//! a weight converted at load has them measured on every call by [`dynamic_scales`], within the
//! forward and its graphs, with no host read between.
//!
//! Every product accumulates in FP32 and rounds once into its input's element type; a bias
//! joins the accumulator, except the packed product's, which joins the rounded product.

use ops::dynamic_scale::{self, DynamicScale, FLOOR};
use ops::fp4::{BLOCK_ELEMENTS, GemmArgs, GemmDescriptor, Nvfp4Gemm, PACKED_M_CLASSES};
use ops::fp8::{
    E4M3_MAX, Fp8Gemm, Fp8GemmArgs, Fp8GemmDescriptor, Fp8Quantize, Fp8QuantizeArgs,
    Fp8QuantizeDescriptor,
};
use ops::gemm::{Args, Descriptor, Gemm};
use ops::reduce::{self, Function, Reduce};
use tensor::{D, DType, Device, Tensor};

use super::nvfp4::{decoded, encoded, launch_quantize};
use super::{ACCUMULATE, Fused, as_rows, columns, compute, like_rows, vector};
use crate::error::{Error, Result};

pub use ops::gemm::WeightOrder;

/// `x · weightᵀ + bias` over a dense weight: `[..., n]` in `x`'s element type for `x` of
/// `[..., k]`.
///
/// `weight` is `[n, k]` in [`WeightOrder::NxK`], as a checkpoint stores a projection, or
/// `[k, n]` in [`WeightOrder::KxN`]; it has `x`'s element type. `bias`, when given, is `[n]`.
///
/// # Errors
///
/// When the weight is not a matrix whose contraction axis is `x`'s row width, the bias is
/// another width, or the operator refuses its arguments.
pub fn linear(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    order: WeightOrder,
) -> Result<Tensor> {
    linear_to(x, weight, bias, order, x.dtype())
}

/// [`linear`] stated in `output`: the product accumulated in FP32 and rounded once into `output`,
/// which may be wider than the operands (a router's FP32 logits of BF16 rows and weight, whose
/// products FP32 holds exactly).
///
/// # Errors
///
/// What [`linear`] refuses.
pub fn linear_to(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    order: WeightOrder,
    output: DType,
) -> Result<Tensor> {
    const OWNER: &str = LinearCall::OWNER;

    let k = columns(OWNER, x)?;
    let &[first, second] = weight.dims() else {
        return Err(Error::shape(
            OWNER,
            format!("a weight is a matrix, not {:?}", weight.dims()),
        ));
    };
    let (contraction, n) = match order {
        WeightOrder::KxN => (first, second),
        WeightOrder::NxK => (second, first),
    };
    if contraction != k as usize {
        return Err(Error::shape(
            OWNER,
            format!(
                "a weight of {:?} in {order} order has another contraction axis than rows of {k}",
                weight.dims()
            ),
        ));
    }
    let n = u32::try_from(n).map_err(|_| Error::shape(OWNER, format!("{n} output channels")))?;
    if let Some(bias) = bias {
        vector(OWNER, "bias", bias, n)?;
    }

    let rows = as_rows(OWNER, x)?;

    compute(LinearCall {
        x,
        rows,
        weight,
        bias,
        order,
        n,
        output,
    })
}

/// A checked call of [`linear_to`]: `rows` is `x` as its rows, `n` the output channels and
/// `output` their element type.
struct LinearCall<'t> {
    x: &'t Tensor,
    rows: Tensor,
    weight: &'t Tensor,
    bias: Option<&'t Tensor>,
    order: WeightOrder,
    n: u32,
    output: DType,
}

impl LinearCall<'_> {
    const OWNER: &'static str = "linear";
}

impl Fused for LinearCall<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.x.device()
    }

    fn cpu(self) -> Result<Self::Output> {
        let Self {
            x,
            rows,
            weight,
            bias,
            order,
            output,
            ..
        } = self;
        let weight = weight.to_dtype(ACCUMULATE)?;
        let weight = match order {
            WeightOrder::KxN => weight,
            WeightOrder::NxK => weight.t()?,
        };

        let mut product = rows.to_dtype(ACCUMULATE)?.matmul(&weight)?;
        if let Some(bias) = bias {
            product = product.broadcast_add(&bias.to_dtype(ACCUMULATE)?)?;
        }
        like_rows(product.to_dtype(output)?, x)
    }

    fn launch(self) -> Result<Self::Output> {
        let Self {
            x,
            rows,
            weight,
            bias,
            order,
            n,
            output,
        } = self;
        let out = ops::empty((rows.dims()[0], n as usize), output)?;
        let args = Args {
            input: ops::operand(&rows)?,
            weight: ops::operand(weight)?,
            bias: bias.map(ops::operand).transpose()?,
            out: ops::operand(&out)?,
            alpha: 1.0,
        };
        ops::launch::<Gemm>(&Descriptor::of(&args, order)?, args)?;
        like_rows(out, x)
    }
}

/// `x · weightᵀ + bias` over a per-tensor FP8 weight:
///
/// ```text
/// q   = e4m3(clamp(x · encode, ±448))                  round to nearest even
/// out = (Σ_k q[·, k] · weight[·, k]) · alpha + bias    alpha = input_scale · weight_scale
/// ```
///
/// `weight` is `[n, k]` E4M3 and `x` a 16-bit activation of `[..., k]`; `encode`, the reciprocal
/// of the activation's scale, and `alpha` are `[1]` F32 tensors: a calibrated export's static
/// factors, or the ones [`dynamic_scales`] measured. The activation is encoded once into an
/// E4M3 buffer of its rows, which the product reads as stored codes. The result is `[..., n]` in
/// `x`'s element type.
///
/// # Errors
///
/// When the weight is not an E4M3 matrix over `x`'s rows, the bias is another width, a factor
/// is not one FP32 number, or the operator refuses its arguments.
pub fn linear_fp8(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    encode: &Tensor,
    alpha: &Tensor,
) -> Result<Tensor> {
    const OWNER: &str = LinearFp8Call::OWNER;

    let k = columns(OWNER, x)?;
    let &[n, contraction] = weight.dims() else {
        return Err(Error::shape(
            OWNER,
            format!("a weight is a matrix, not {:?}", weight.dims()),
        ));
    };
    if contraction != k as usize || weight.dtype() != DType::F8E4M3 {
        return Err(Error::shape(
            OWNER,
            format!(
                "an FP8 weight is [n, {k}] F8_E4M3, not {} {:?}",
                ops::abi::name(weight.dtype()),
                weight.dims()
            ),
        ));
    }
    let n = u32::try_from(n).map_err(|_| Error::shape(OWNER, format!("{n} output channels")))?;
    if let Some(bias) = bias {
        vector(OWNER, "bias", bias, n)?;
    }
    factor(OWNER, "encode", encode)?;
    factor(OWNER, "alpha", alpha)?;

    let rows = as_rows(OWNER, x)?;

    compute(LinearFp8Call {
        x,
        rows,
        weight,
        bias,
        encode,
        alpha,
        n,
    })
}

/// A checked call of [`linear_fp8`]: `rows` is `x` as its rows, and `n` the output channels.
struct LinearFp8Call<'t> {
    x: &'t Tensor,
    rows: Tensor,
    weight: &'t Tensor,
    bias: Option<&'t Tensor>,
    encode: &'t Tensor,
    alpha: &'t Tensor,
    n: u32,
}

impl LinearFp8Call<'_> {
    const OWNER: &'static str = "linear_fp8";
}

impl Fused for LinearFp8Call<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.x.device()
    }

    fn cpu(self) -> Result<Self::Output> {
        let Self {
            x,
            rows,
            weight,
            bias,
            encode,
            alpha,
            ..
        } = self;
        let limit = f64::from(E4M3_MAX);
        let encoded = rows
            .to_dtype(ACCUMULATE)?
            .broadcast_mul(&encode.reshape(())?)?
            .clamp(-limit, limit)?
            .to_dtype(DType::F8E4M3)?
            .to_dtype(ACCUMULATE)?;

        let mut product = encoded
            .matmul(&weight.to_dtype(ACCUMULATE)?.t()?)?
            .broadcast_mul(&alpha.reshape(())?)?;
        if let Some(bias) = bias {
            product = product.broadcast_add(&bias.to_dtype(ACCUMULATE)?)?;
        }
        like_rows(product.to_dtype(x.dtype())?, x)
    }

    fn launch(self) -> Result<Self::Output> {
        let Self {
            x,
            rows,
            weight,
            bias,
            encode,
            alpha,
            n,
        } = self;
        // 1. The activation's codes, each row encoded under the one multiplier.
        let (m, k) = (rows.dims()[0], rows.dims()[1]);
        let codes = ops::empty((m, k), DType::F8E4M3)?;
        ops::launch::<Fp8Quantize>(
            &Fp8QuantizeDescriptor::new(x.dtype()),
            Fp8QuantizeArgs {
                input: ops::operand(&rows)?,
                out: ops::operand(&codes)?,
                encode: ops::operand(encode)?,
            },
        )?;

        // 2. The product over the two encodings.
        let out = ops::empty((m, n as usize), x.dtype())?;
        ops::launch::<Fp8Gemm>(
            &Fp8GemmDescriptor::new(x.dtype(), bias.is_some()),
            Fp8GemmArgs {
                input: ops::operand(&codes)?,
                weight: ops::operand(weight)?,
                bias: bias.map(ops::operand).transpose()?,
                out: ops::operand(&out)?,
                alpha: ops::operand(alpha)?,
            },
        )?;
        like_rows(out, x)
    }
}

/// The two factors a product over `x` encoded at its own measured scale reads, against a weight
/// whose largest magnitude is `weight_amax`: `(encode, alpha)`, `[1]` F32 each.
///
/// ```text
/// amax   = max(max |x|, 1e-12)                          over every element of x
/// encode = range / amax                                 the encoder's multiplier
/// alpha  = (amax / range) · (weight_amax / range)       the product's factor
/// ```
///
/// `range` is the format's ([`ops::dynamic_scale::FP8_RANGE`], [`NVFP4_RANGE`]); the divisions
/// round to nearest. `x` is a BF16, FP16 or FP32 activation of `[..., k]`, read as its rows, and
/// `weight_amax` one FP32 number. On a device, each row's largest magnitude is folded by
/// `reduce` (`absmax`) and the rows' by `dynamic_scale`: two launches whose outputs come from the
/// context's allocator and which read nothing back, so a captured forward replays them.
///
/// [`NVFP4_RANGE`]: ops::dynamic_scale::NVFP4_RANGE
///
/// # Errors
///
/// When `x` is not floating or has no element, `weight_amax` is not one FP32 number, or an
/// operator refuses its arguments.
pub fn dynamic_scales(x: &Tensor, weight_amax: &Tensor, range: f32) -> Result<(Tensor, Tensor)> {
    const OWNER: &str = DynamicScalesCall::OWNER;

    if !matches!(x.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        return Err(Error::shape(
            OWNER,
            format!(
                "{} activations; a scale is measured from BF16, FP16 or FP32",
                ops::abi::name(x.dtype())
            ),
        ));
    }
    factor(OWNER, "weight's largest magnitude", weight_amax)?;
    let rows = as_rows(OWNER, x)?;
    let count = rows.dims()[0];
    if count == 0 {
        return Err(Error::shape(
            OWNER,
            "an activation of no rows has no magnitude".to_owned(),
        ));
    }

    compute(DynamicScalesCall {
        x,
        rows,
        weight_amax,
        range,
        count,
    })
}

/// A checked call of [`dynamic_scales`]: `rows` is `x` as its `count` rows.
struct DynamicScalesCall<'t> {
    x: &'t Tensor,
    rows: Tensor,
    weight_amax: &'t Tensor,
    range: f32,
    count: usize,
}

impl DynamicScalesCall<'_> {
    const OWNER: &'static str = "dynamic_scales";
}

impl Fused for DynamicScalesCall<'_> {
    type Output = (Tensor, Tensor);

    fn device(&self) -> &Device {
        self.x.device()
    }

    fn cpu(self) -> Result<Self::Output> {
        let Self {
            rows,
            weight_amax,
            range,
            ..
        } = self;
        let amax = rows
            .to_dtype(ACCUMULATE)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?
            .max(FLOOR);
        let weight = weight_amax
            .to_dtype(ACCUMULATE)?
            .flatten_all()?
            .to_vec1::<f32>()?[0];

        let encode = Tensor::new(&[range / amax], &Device::Cpu)?;
        let alpha = Tensor::new(&[(amax / range) * (weight / range)], &Device::Cpu)?;
        Ok((encode, alpha))
    }

    fn launch(self) -> Result<Self::Output> {
        let Self {
            x,
            rows,
            weight_amax,
            range,
            count,
        } = self;
        // 1. Each row's largest magnitude, in the activation's own type: a magnitude is exact. The
        //    fold is the tensor primitives' own reduction, so every projection's rows are folded by
        //    the module of one key.
        let partials = ops::empty((count, 1), x.dtype())?;
        let fold = reduce::Args {
            x: ops::operand(&rows)?,
            out: ops::operand(&partials)?,
            axes: &[1],
        };
        ops::launch::<Reduce>(&reduce::Descriptor::of(Function::AbsMax, &fold), fold)?;

        // 2. The rows' largest, and the two factors from it and the weight's.
        let encode = ops::empty(1, DType::F32)?;
        let alpha = ops::empty(1, DType::F32)?;
        ops::launch::<DynamicScale>(
            &dynamic_scale::Descriptor::new(x.dtype()),
            dynamic_scale::Args {
                partials: ops::operand(&partials.flatten_all()?)?,
                weight_amax: ops::operand(weight_amax)?,
                encode: ops::operand(&encode)?,
                alpha: ops::operand(&alpha)?,
                range,
                floor: FLOOR,
            },
        )?;
        Ok((encode, alpha))
    }
}

/// Refuses `tensor`, the factor `what` of an encoded product, unless it is one FP32 number.
fn factor(owner: &'static str, what: &str, tensor: &Tensor) -> Result<()> {
    if tensor.dtype() != DType::F32 || tensor.elem_count() != 1 {
        return Err(Error::shape(
            owner,
            format!(
                "the {what} is {} {:?}, and a product reads one FP32 number",
                ops::abi::name(tensor.dtype()),
                tensor.dims()
            ),
        ));
    }
    Ok(())
}

/// `x · weightᵀ + bias` over an NVFP4 weight: the activation is encoded into the weight's own
/// representation and the product is taken over the two encodings.
///
/// ```text
/// encode   scale[row, block] = e4m3(amax(block) · input_scale / 6)
///          code[row, element] = e2m1(x / (scale · (1 / input_scale)))
/// product  p   = round((Σ_k code_x · scale_x · code_w · scale_w) · alpha)
/// bias     out = round(p + bias)
/// ```
///
/// `values` is the packed weight, `[n, k / 2]` U8 with two E2M1 codes a byte, the even element
/// in the low nibble; `scales` its E4M3 block scales, one per sixteen elements of a row, in the
/// swizzled 128x4 layout the product reads; `input_scale` the `[1]` FP32 reciprocal of the
/// activation's tensor scale, the encoder's global scale; `alpha` the `[1]` FP32 product of the
/// weight's tensor scale and the activation's. Both are a calibrated export's constants, or the
/// factors [`dynamic_scales`] measured from `x`. `x` is a BF16 or FP16 activation of
/// `[..., k]`, and the result is `[..., n]` in its element type.
///
/// The packed product has no bias epilogue, so `bias`, `[n]` when given, joins the product
/// after it has rounded into the activation's type, and the sum rounds again: one rounding
/// more than a dense product, whose bias joins the FP32 accumulator.
///
/// The packed product is compiled per row class ([`ops::fp4::PACKED_M_CLASSES`]) and writes
/// whole tiles. Rows up to the largest class are one launch at the smallest class that holds
/// them; more rows are computed block by block, every block the largest class's rows and the
/// last one the rest at its own class. Each block's buffers are taken at its class, and the
/// result is the leading rows of the blocks' products.
///
/// The CPU formula divides where the device's encoder multiplies by the hardware's approximate
/// reciprocal, so a code that lies on a rounding boundary may differ from the device's by one
/// step; a device check compares the two under the operators' tolerance.
///
/// # Errors
///
/// When a field is not the element type or the extents the weight's rows and columns take, the
/// bias is another width, the activation is not BF16 or FP16 or has no rows, or an operator
/// refuses its arguments.
pub fn linear_nvfp4(
    x: &Tensor,
    values: &Tensor,
    scales: &Tensor,
    bias: Option<&Tensor>,
    input_scale: &Tensor,
    alpha: &Tensor,
) -> Result<Tensor> {
    const OWNER: &str = "linear_nvfp4";
    let refuse = |problem: String| Error::shape(OWNER, problem);

    let &[n, packed] = values.dims() else {
        return Err(refuse(format!(
            "a packed weight is a U8 matrix, not {:?}",
            values.dims()
        )));
    };
    if values.dtype() != DType::U8 {
        return Err(refuse(format!(
            "a packed weight is a U8 matrix, not {} {:?}",
            ops::abi::name(values.dtype()),
            values.dims()
        )));
    }
    let k = columns(OWNER, x)?;
    if packed * 2 != k as usize || !k.is_multiple_of(BLOCK_ELEMENTS) {
        return Err(refuse(format!(
            "rows of {k} elements against a packed weight of {:?}: the contraction is whole \
             sixteen-element blocks, two codes a byte",
            values.dims()
        )));
    }

    // One layout governs a product: it states both how the activation's block scales are
    // arranged and, by naming the product, how the weight's are read.
    let expected = GemmDescriptor::SCALE_LAYOUT.shape(n as u64, u64::from(k));
    let stated: Vec<u64> = scales.dims().iter().map(|dim| *dim as u64).collect();
    if stated != expected {
        return Err(refuse(format!(
            "the block scales are {:?}; {n} rows of {k} elements take {:?} in the {} layout",
            scales.dims(),
            expected,
            GemmDescriptor::SCALE_LAYOUT
        )));
    }
    factor(OWNER, "input scale", input_scale)?;
    factor(OWNER, "alpha", alpha)?;

    // The encoder reads the activation's own type, which NVFP4 encodes from BF16 or FP16.
    if !matches!(x.dtype(), DType::BF16 | DType::F16) {
        return Err(refuse(format!(
            "{} activations; NVFP4 encodes BF16 or FP16",
            ops::abi::name(x.dtype())
        )));
    }
    let outputs = u32::try_from(n).map_err(|_| refuse(format!("{n} output channels")))?;
    if let Some(bias) = bias {
        vector(OWNER, "bias", bias, outputs)?;
    }
    let rows = as_rows(OWNER, x)?;
    let count = rows.dims()[0];
    if count == 0 {
        return Err(refuse("a projection of no rows has no kernel".to_owned()));
    }

    let product = compute(LinearNvfp4Call {
        x,
        rows,
        values,
        scales,
        input_scale,
        alpha,
    })?;

    // The bias joins the rounded product in the activation's type.
    let out = match bias {
        Some(bias) => product.broadcast_add(&bias.to_dtype(x.dtype())?)?,
        None => product,
    };
    like_rows(out, x)
}

/// A checked call of [`linear_nvfp4`]'s product, before its bias: `rows` is `x` as its rows.
struct LinearNvfp4Call<'t> {
    x: &'t Tensor,
    rows: Tensor,
    values: &'t Tensor,
    scales: &'t Tensor,
    input_scale: &'t Tensor,
    alpha: &'t Tensor,
}

impl Fused for LinearNvfp4Call<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.x.device()
    }

    fn cpu(self) -> Result<Self::Output> {
        let Self {
            x,
            rows,
            values,
            scales,
            input_scale,
            alpha,
        } = self;
        Ok(nvfp4_formula(&rows, values, scales, input_scale, alpha)?.to_dtype(x.dtype())?)
    }

    fn launch(self) -> Result<Self::Output> {
        let Self {
            rows,
            values,
            scales,
            input_scale,
            alpha,
            ..
        } = self;
        packed_product(&rows, values, scales, input_scale, alpha)
    }
}

/// The packed product of [`linear_nvfp4`] on a device: `rows`, the `[m, k]` activation, encoded
/// and multiplied by the packed weight block by block, `[m, n]` in the activation's type.
fn packed_product(
    rows: &Tensor,
    values: &Tensor,
    scales: &Tensor,
    input_scale: &Tensor,
    alpha: &Tensor,
) -> Result<Tensor> {
    let (count, k) = (rows.dims()[0], rows.dims()[1]);
    let (n, packed) = (values.dims()[0], values.dims()[1]);

    // The blocks the rows are computed in, `(first row, rows, class)`: the largest class's rows
    // each, and the rest at the smallest class that holds them.
    let largest = PACKED_M_CLASSES[PACKED_M_CLASSES.len() - 1] as usize;
    let mut blocks = Vec::with_capacity(count.div_ceil(largest));
    let mut first = 0;
    while first < count {
        let held = (count - first).min(largest);
        let class = ops::fp4::packed_rows(held as u32)? as usize;

        blocks.push((first, held, class));
        first += held;
    }

    // A product writes its class's rows. Every block but the last holds exactly its class, so
    // the products laid end to end are the rows in order, and only the last block's rows past
    // the activation's trail them.
    let padded: usize = blocks.iter().map(|(_, _, class)| class).sum();
    let out = ops::empty((padded, n), rows.dtype())?;

    for (first, held, class) in blocks {
        // 1. Encode the block's rows into the layout the product reads.
        //
        // The buffers are class-height and the encode writes the block's rows into their first
        // rows; the product's rows past those are computed from bytes nothing reads and are not
        // part of the result.
        let packed_rows = ops::empty((class, packed), DType::U8)?;
        let class_scales = GemmDescriptor::SCALE_LAYOUT.shape(class as u64, k as u64);
        let packed_scales = ops::empty(
            (class_scales[0] as usize, class_scales[1] as usize),
            DType::U8,
        )?;
        let token_scale = ops::empty(class, DType::F32)?;

        let row_scales = GemmDescriptor::SCALE_LAYOUT.shape(held as u64, k as u64);
        launch_quantize(
            &rows.narrow(0, first, held)?,
            input_scale,
            GemmDescriptor::SCALE_LAYOUT,
            (
                ops::operand(&packed_rows.narrow(0, 0, held)?)?,
                ops::operand(&packed_scales.narrow(0, 0, row_scales[0] as usize)?)?,
                // One scalar per row, so the tensor has one axis.
                ops::operand(&token_scale.narrow(0, 0, held)?)?,
            ),
        )?;

        // 2. Multiply the encoded rows by the weight, over the class's rows.
        let descriptor = GemmDescriptor::new(
            scales.dtype(),
            rows.dtype(),
            class as u32,
            n as u32,
            k as u32,
        );

        // The scale's type is the checkpoint's own statement, so scales declared as anything
        // but E4M3 make a key no store built: refused here, where the function can say which
        // tensor.
        ops::fp4::check_gemm(&descriptor)?;

        ops::launch::<Nvfp4Gemm>(
            &descriptor,
            GemmArgs {
                input: ops::operand(&packed_rows)?,
                weight: ops::operand(values)?,
                input_scales: ops::operand(&packed_scales)?,
                weight_scales: ops::operand(scales)?,
                alpha: ops::operand(alpha)?,
                bias: None,
                out: ops::operand(&out.narrow(0, first, class)?)?,
            },
        )?;
    }
    Ok(out.narrow(0, 0, count)?)
}

/// The FP32 product of [`linear_nvfp4`] on CPU tensors, before its one rounding: `rows` is the
/// `[m, k]` activation.
fn nvfp4_formula(
    rows: &Tensor,
    values: &Tensor,
    scales: &Tensor,
    input_scale: &Tensor,
    alpha: &Tensor,
) -> Result<Tensor> {
    let (m, k) = (rows.dims()[0], rows.dims()[1]);

    // The activation, encoded and decoded: each element's E2M1 code times its block's E4M3
    // scale. The row factor and the two tensor scales are one number, `alpha`.
    let (codes, scale) = encoded(rows, input_scale)?;
    let activation = codes
        .broadcast_mul(&scale.unsqueeze(D::Minus1)?)?
        .reshape((m, k))?;

    // The weight, decoded: its codes times its block scales, read out of the swizzled table.
    let weight = decoded(values, scales, GemmDescriptor::SCALE_LAYOUT)?;

    Ok(activation
        .matmul(&weight.t()?)?
        .broadcast_mul(&alpha.reshape(())?)?)
}

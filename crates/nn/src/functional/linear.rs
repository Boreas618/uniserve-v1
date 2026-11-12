//! Linear projections `out = x · weightᵀ + bias`, one function per representation a checkpoint
//! stores a weight in.
//!
//! ```text
//! linear        a dense weight: one `gemm`
//! linear_fp8    a per-tensor FP8 weight: one `fp8_gemm`, which encodes the activation by its
//!               static scale as it loads it
//! linear_nvfp4  a calibrated NVFP4 weight: the activation encoded (`nvfp4_quantize`) and the
//!               packed product (`nvfp4_gemm`)
//! ```
//!
//! Every product accumulates in FP32 and rounds once into its input's element type.

use ops::fp4::{
    BLOCK_ELEMENTS, GemmArgs, GemmDescriptor, Nvfp4Gemm, Nvfp4Quantize, QuantizeArgs,
    QuantizeDescriptor, ScaleLayout,
};
use ops::fp8::{E4M3_MAX, Fp8Gemm, Fp8GemmArgs, Fp8GemmDescriptor};
use ops::gemm::{Args, Descriptor, Epilogue, Gemm};
use uniserve_tensor::{D, DType, Device, Tensor};

use super::{ACCUMULATE, as_rows, columns, like_rows, on_cpu, vector};
use crate::error::{Error, Result};
use crate::quantization::e4m3;

pub use ops::gemm::WeightOrder;

/// Elements of a row one activation-encoding kernel covers.
///
/// A whole number of sixteen-element scale blocks, as the layout requires, and a key field of
/// `nvfp4_quantize`: every packed projection and expert block states this one value, so one
/// launch site compiles one kernel.
pub const ENCODE_CHUNK: u32 = 1024;

/// The encoder's chunk is a whole number of scale blocks; a chunk that were not would have a
/// partial block with no scale to name it.
const _: () = assert!(ENCODE_CHUNK.is_multiple_of(BLOCK_ELEMENTS));

/// The arrangement the packed product reads block scales in, the weight's and the encoded
/// activation's alike.
pub(crate) const PRODUCT_SCALE_LAYOUT: ScaleLayout = ScaleLayout::Swizzled128X4;

/// The tile an `fp8_gemm` kernel covers, `(rows, output columns, contraction elements)`: one
/// kernel set serves every projection.
const FP8_TILE: (u32, u32, u32) = (64, 128, 128);

/// The tile one encoding of a `columns`-wide row runs in.
///
/// The package refuses a tile that does not divide the row, so the requested tile is halved
/// until it divides. A row narrower than the largest tile, such as the mixture-of-experts'
/// 768-element second encode, gets the largest tile that fits it instead of a key no package
/// can build.
pub fn encode_block(columns: u32, requested: u32) -> u32 {
    let mut block = requested.max(1);
    while block > 1 && !columns.is_multiple_of(block) {
        block /= 2;
    }
    block
}

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
    const OWNER: &str = "linear";

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
    if on_cpu(x) {
        let weight = weight.to_dtype(ACCUMULATE)?;
        let weight = match order {
            WeightOrder::KxN => weight,
            WeightOrder::NxK => weight.t()?,
        };

        let mut product = rows.to_dtype(ACCUMULATE)?.matmul(&weight)?;
        if let Some(bias) = bias {
            product = product.broadcast_add(&bias.to_dtype(ACCUMULATE)?)?;
        }
        return like_rows(product.to_dtype(x.dtype())?, x);
    }

    let out = backend::empty((rows.dims()[0], n as usize), x.dtype())?;
    backend::launch::<Gemm>(
        &Descriptor {
            input: x.dtype(),
            weight: weight.dtype(),
            accumulate: ACCUMULATE,
            output: x.dtype(),
            order,
            epilogue: epilogue(bias),
            k,
        },
        Args {
            input: backend::operand(&rows)?,
            weight: backend::operand(weight)?,
            bias: bias.map(backend::operand).transpose()?,
            out: backend::operand(&out)?,
            alpha: 1.0,
        },
    )?;
    like_rows(out, x)
}

/// The epilogue a bias asks for.
fn epilogue(bias: Option<&Tensor>) -> Epilogue {
    if bias.is_some() {
        Epilogue::Bias
    } else {
        Epilogue::None
    }
}

/// `x · weightᵀ + bias` over a per-tensor FP8 weight (ModelOpt's `FP8`):
///
/// ```text
/// q   = e4m3(clamp(x · (1 / input_scale), ±448))      round to nearest even
/// out = (Σ_k q[·, k] · weight[·, k]) · alpha + bias    alpha = input_scale · weight_scale
/// ```
///
/// `weight` is `[n, k]` E4M3 and `x` a 16-bit activation of `[..., k]`; the encoding is fused
/// into the product's load, so the activation is never stored in E4M3. The result is `[..., n]`
/// in `x`'s element type.
///
/// # Errors
///
/// When the weight is not an E4M3 matrix over `x`'s rows, the bias is another width, the scales
/// are not positive finite numbers, or the operator refuses its arguments.
pub fn linear_fp8(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    input_scale: f32,
    alpha: f32,
) -> Result<Tensor> {
    const OWNER: &str = "linear_fp8";

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
    if !(input_scale.is_finite() && input_scale > 0.0 && alpha.is_finite()) {
        return Err(Error::shape(
            OWNER,
            format!("an input scale of {input_scale} and a factor of {alpha} dequantize nothing"),
        ));
    }

    let rows = as_rows(OWNER, x)?;
    if on_cpu(x) {
        let limit = f64::from(E4M3_MAX);
        let encoded = rows
            .to_dtype(ACCUMULATE)?
            .affine(f64::from(1.0 / input_scale), 0.0)?
            .clamp(-limit, limit)?
            .to_dtype(DType::F8E4M3)?
            .to_dtype(ACCUMULATE)?;

        let mut product = encoded
            .matmul(&weight.to_dtype(ACCUMULATE)?.t()?)?
            .affine(f64::from(alpha), 0.0)?;
        if let Some(bias) = bias {
            product = product.broadcast_add(&bias.to_dtype(ACCUMULATE)?)?;
        }
        return like_rows(product.to_dtype(x.dtype())?, x);
    }

    let out = backend::empty((rows.dims()[0], n as usize), x.dtype())?;
    let (block_m, block_n, block_k) = FP8_TILE;
    backend::launch::<Fp8Gemm>(
        &Fp8GemmDescriptor {
            dtype: x.dtype(),
            block_m,
            block_n,
            block_k,
            bias: bias.is_some(),
        },
        Fp8GemmArgs {
            input: backend::operand(&rows)?,
            weight: backend::operand(weight)?,
            bias: bias.map(backend::operand).transpose()?,
            out: backend::operand(&out)?,
            input_scale,
            alpha,
        },
    )?;
    like_rows(out, x)
}

/// `x · weightᵀ` over a calibrated NVFP4 weight: the activation is encoded into the weight's
/// own representation and the product is taken over the two encodings.
///
/// ```text
/// encode   scale[row, block] = e4m3(amax(block) · input_scale / 6)
///          code[row, element] = e2m1(x / (scale · (1 / input_scale)))
/// product  out = (Σ_k code_x · scale_x · code_w · scale_w) · alpha
/// ```
///
/// `values` is the packed weight, `[n, k / 2]` U8 with two E2M1 codes a byte, the even element
/// in the low nibble; `scales` its E4M3 block scales, one per sixteen elements of a row, in the
/// swizzled 128x4 layout the product reads; `input_scale` the `[1]` FP32 reciprocal of the
/// calibration's activation scale, the encoder's global scale; `alpha` the `[1]` FP32 product of
/// the weight's tensor scale and the activation's. `x` is a BF16 or FP16 activation of
/// `[..., k]`, and the result is `[..., n]` in its element type.
///
/// The packed product is compiled per row class ([`ops::fp4::packed_rows`]) and writes whole
/// tiles, so its buffers are taken at the class of `x`'s rows and the result is the first rows
/// of the product's block.
///
/// The CPU formula divides where the device's encoder multiplies by the hardware's approximate
/// reciprocal, so a code that lies on a rounding boundary may differ from the device's by one
/// step; a device check compares the two under the operators' tolerance.
///
/// # Errors
///
/// When a field is not the element type or the extents the weight's rows and columns take, the
/// activation is not BF16 or FP16 or has no rows, the row count is above the largest packed
/// class, or an operator refuses its arguments.
pub fn linear_nvfp4(
    x: &Tensor,
    values: &Tensor,
    scales: &Tensor,
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
    let expected = PRODUCT_SCALE_LAYOUT.shape(n as u64, u64::from(k));
    let stated: Vec<u64> = scales.dims().iter().map(|dim| *dim as u64).collect();
    if stated != expected.dims() {
        return Err(refuse(format!(
            "the block scales are {:?}; {n} rows of {k} elements take {:?} in the \
             {PRODUCT_SCALE_LAYOUT} layout",
            scales.dims(),
            expected.dims()
        )));
    }
    for (what, scalar) in [("input scale", input_scale), ("alpha", alpha)] {
        if scalar.dtype() != DType::F32 || scalar.elem_count() != 1 {
            return Err(refuse(format!(
                "the {what} is {} {:?}, and the product reads one FP32 number",
                ops::abi::name(scalar.dtype()),
                scalar.dims()
            )));
        }
    }

    // The encoder reads the activation's own type, which NVFP4 encodes from BF16 or FP16.
    if !matches!(x.dtype(), DType::BF16 | DType::F16) {
        return Err(refuse(format!(
            "{} activations; NVFP4 encodes BF16 or FP16",
            ops::abi::name(x.dtype())
        )));
    }
    let rows = as_rows(OWNER, x)?;
    let count = rows.dims()[0];
    if count == 0 {
        return Err(refuse("a projection of no rows has no kernel".to_owned()));
    }

    if on_cpu(x) {
        let product = nvfp4_formula(&rows, values, scales, input_scale, alpha)?;
        return like_rows(product.to_dtype(x.dtype())?, x);
    }

    let counted = u32::try_from(count).map_err(|_| refuse(format!("{count} activation rows")))?;
    let class = ops::fp4::packed_rows(counted)? as usize;

    // 1. Encode the activation into the layout the product reads.
    //
    // The buffers are class-height and the encode writes the activation's rows into their
    // first rows; the product's rows past those are computed from bytes nothing reads and are
    // not part of the result.
    let encoded = backend::empty((class, packed), DType::U8)?;
    let class_scales = PRODUCT_SCALE_LAYOUT.shape(class as u64, u64::from(k));
    let encoded_scales = backend::empty(
        (class_scales.dim(0) as usize, class_scales.dim(1) as usize),
        DType::U8,
    )?;
    let token_scale = backend::empty(class, DType::F32)?;

    let row_scales = PRODUCT_SCALE_LAYOUT.shape(count as u64, u64::from(k));
    backend::launch::<Nvfp4Quantize>(
        &QuantizeDescriptor {
            input: x.dtype(),
            columns: k,
            layout: PRODUCT_SCALE_LAYOUT,
            block: encode_block(k, ENCODE_CHUNK),
            pdl: false,
        },
        QuantizeArgs {
            input: backend::operand(&rows)?,
            values: backend::operand(&encoded.narrow(0, 0, count)?)?,
            scales: backend::operand(&encoded_scales.narrow(0, 0, row_scales.dim(0) as usize)?)?,
            global_scale: backend::operand(input_scale)?,
            // One scalar per row, so the tensor has one axis.
            token_scale: backend::operand(&token_scale.narrow(0, 0, count)?)?,
        },
    )?;

    // 2. Multiply the encoded activation by the weight, over the class's rows.
    let descriptor = GemmDescriptor {
        packed: DType::U8,
        scale: scales.dtype(),
        accumulate: ACCUMULATE,
        output: x.dtype(),
        layout: PRODUCT_SCALE_LAYOUT,
        epilogue: Epilogue::None,
        m: class as u32,
        n: u32::try_from(n).map_err(|_| refuse(format!("{n} output channels")))?,
        k,
    };

    // The scale's type is the checkpoint's own statement, so scales declared as anything but
    // E4M3 make a key no store built: refused here, where the function can say which tensor.
    ops::fp4::check_gemm(&descriptor)?;

    let out = backend::empty((class, n), x.dtype())?;
    backend::launch::<Nvfp4Gemm>(
        &descriptor,
        GemmArgs {
            input: backend::operand(&encoded)?,
            weight: backend::operand(values)?,
            input_scales: backend::operand(&encoded_scales)?,
            weight_scales: backend::operand(scales)?,
            alpha: backend::operand(alpha)?,
            bias: None,
            out: backend::operand(&out)?,
        },
    )?;
    like_rows(out.narrow(0, 0, count)?, x)
}

/// The eight magnitudes an E2M1 code states, then the same negated: a code's fourth bit is its
/// sign.
const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

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
    let blocks = k / BLOCK_ELEMENTS as usize;
    let global = input_scale.reshape(())?;

    // The activation, encoded and decoded: each block's E4M3 scale under the global scale, and
    // each element's E2M1 code relative to it.
    let x = rows.to_dtype(ACCUMULATE)?.reshape((m, blocks, 16))?;
    let amax = x.abs()?.max(D::Minus1)?;
    let scale = (amax / 6.0)?
        .broadcast_mul(&global)?
        .to_dtype(DType::F8E4M3)?
        .to_dtype(ACCUMULATE)?;

    // A block whose scale rounds to zero encodes to zeros: the encoder multiplies by zero
    // there, where the reciprocal of a zero scale is not a number.
    let token = global.recip()?;
    let relative = scale.broadcast_mul(&token)?;
    let inverse = relative
        .ne(0.0)?
        .where_cond(&relative.recip()?, &relative.zeros_like()?)?;
    let codes = e2m1(&x.broadcast_mul(&inverse.unsqueeze(D::Minus1)?)?)?;
    let activation = codes
        .broadcast_mul(&scale.unsqueeze(D::Minus1)?)?
        .reshape((m, k))?;

    // The weight, decoded: its codes times its block scales, read out of the swizzled table.
    let n = values.dims()[0];
    let high = values.broadcast_div(&integer(16, values)?)?;
    let low = remainder(values, 16)?;
    let nibbles = Tensor::stack(&[low, high], D::Minus1)?.flatten_all()?;
    let weight_codes = Tensor::new(&E2M1, &Device::Cpu)?
        .index_select(&nibbles, 0)?
        .reshape((n, blocks, 16))?;

    let padded_columns = scales.dims()[1];
    let weight_scales = e4m3_values(
        &scales
            .flatten_all()?
            .index_select(&swizzled_offsets(n, blocks, padded_columns)?, 0)?,
    )?
    .reshape((n, blocks, 1))?;
    let weight = weight_codes
        .broadcast_mul(&weight_scales)?
        .reshape((n, k))?;

    Ok(activation
        .matmul(&weight.t()?)?
        .broadcast_mul(&alpha.reshape(())?)?)
}

/// `x`, FP32, rounded to the nearest E2M1 value, ties to the code whose mantissa bit is zero,
/// saturating at ±6: the eight magnitudes are 0, 0.5, 1, 1.5, 2, 3, 4 and 6.
fn e2m1(x: &Tensor) -> Result<Tensor> {
    let magnitude = x.abs()?;

    // Each step of the grid, by the midpoint below it. A midpoint belongs to the neighbour with
    // the even mantissa, which is the lower value at 0.25, 1.25, 2.5 and 5 and the upper one at
    // 0.75, 1.75 and 3.5.
    let step = |reached: Tensor, height: f64| -> Result<Tensor> {
        Ok(reached.to_dtype(ACCUMULATE)?.affine(height, 0.0)?)
    };
    let steps = [
        step(magnitude.gt(0.25)?, 0.5)?,
        step(magnitude.ge(0.75)?, 0.5)?,
        step(magnitude.gt(1.25)?, 0.5)?,
        step(magnitude.ge(1.75)?, 0.5)?,
        step(magnitude.gt(2.5)?, 1.0)?,
        step(magnitude.ge(3.5)?, 1.0)?,
        step(magnitude.gt(5.0)?, 2.0)?,
    ];

    let mut rounded = magnitude.zeros_like()?;
    for step in &steps {
        rounded = (rounded + step)?;
    }
    Ok((rounded * x.sign()?)?)
}

/// The FP32 values of `scales`, E4M3 block scales stored as `F8_E4M3` or as their bytes.
fn e4m3_values(scales: &Tensor) -> Result<Tensor> {
    if scales.dtype() == DType::F8E4M3 {
        return Ok(scales.to_dtype(ACCUMULATE)?);
    }

    // Raw bytes have no cast: each byte selects its value from the format's 256.
    let table: Vec<f32> = (0..=u8::MAX).map(e4m3).collect();
    Ok(Tensor::new(table.as_slice(), &Device::Cpu)?.index_select(scales, 0)?)
}

/// `value` as a one-element tensor of `like`'s integer element type, for integer arithmetic,
/// which the scalar operators of the tensor library evaluate in floating point.
pub(super) fn integer(value: u32, like: &Tensor) -> Result<Tensor> {
    Ok(Tensor::new(value, &Device::Cpu)?.to_dtype(like.dtype())?)
}

/// `values mod modulus`, elementwise over an integer tensor.
pub(super) fn remainder(values: &Tensor, modulus: u32) -> Result<Tensor> {
    let modulus = integer(modulus, values)?;
    let whole = values.broadcast_div(&modulus)?.broadcast_mul(&modulus)?;

    Ok(values.broadcast_sub(&whole)?)
}

/// The flat offsets, `[rows · blocks]` U32 in row-major order, of a `[rows, blocks]` block-scale
/// matrix inside its swizzled 128x4 table of `padded_columns` scale columns: column groups of
/// four stride 512, thirty-two-row groups stride 16, groups of 128 rows stride
/// `128 · padded_columns`.
fn swizzled_offsets(rows: usize, blocks: usize, padded_columns: usize) -> Result<Tensor> {
    let row = Tensor::arange(0_u32, rows as u32, &Device::Cpu)?;
    let column = Tensor::arange(0_u32, blocks as u32, &Device::Cpu)?;
    let times = |values: Tensor, factor: usize| -> Result<Tensor> {
        Ok(values.broadcast_mul(&integer(factor as u32, &values)?)?)
    };
    let over = |values: &Tensor, divisor: u32| -> Result<Tensor> {
        Ok(values.broadcast_div(&integer(divisor, values)?)?)
    };

    let lane = times(remainder(&row, 32)?, 16)?;
    let group = times(over(&remainder(&row, 128)?, 32)?, 4)?;
    let tile = times(over(&row, 128)?, 128 * padded_columns)?;
    let row_offset = ((lane + group)? + tile)?;

    let column_offset = (remainder(&column, 4)? + times(over(&column, 4)?, 512)?)?;

    Ok(row_offset
        .unsqueeze(1)?
        .broadcast_add(&column_offset.unsqueeze(0)?)?
        .flatten_all()?)
}

//! The NVFP4 encoding of an activation, and the decoding the CPU formulas of the packed products
//! read.
//!
//! A product over a calibrated NVFP4 weight reads its activation in the weight's own
//! representation: packed E2M1 codes, two a byte, with one E4M3 scale per block of sixteen
//! elements and one FP32 factor per row. [`nvfp4_quantize`] is that encoding as a function; the
//! products that encode their own activation ([`linear_nvfp4`](super::linear_nvfp4), the routed
//! expert products of [`super::gather_gemm_nvfp4`] and [`super::route_gemm_nvfp4`]) make the same
//! launch over buffers of their own heights.
//!
//! ```text
//! scale[row, block]   = e4m3(amax(block) · global_scale / 6)          round to nearest even
//! code[row, element]  = e2m1(x / (scale · (1 / global_scale)))        round to nearest even
//! factor[row]         = 1 / global_scale
//! x̂[row, element]     = code · scale · factor
//! ```
//!
//! The device's encoder multiplies by the hardware's approximate reciprocals where the formula
//! divides (`1 / 6`, `1 / global_scale` and `1 / (scale · factor)`), so a code or a scale that
//! lies on a rounding boundary may differ from the formula's by one step; a device check
//! compares the two under the consuming product's tolerance.

use float8::F8E4M3;
use ops::fp4::{BLOCK_ELEMENTS, Nvfp4Quantize, QuantizeArgs, QuantizeDescriptor, ScaleLayout};
use tensor::{D, DType, Device, Tensor};

use super::{ACCUMULATE, PDL, as_rows, columns, on_cpu};
use crate::error::{Error, Result};

/// Elements of a row one activation-encoding kernel covers.
///
/// A whole number of sixteen-element scale blocks, as the layout requires, and a key field of
/// `nvfp4_quantize`: every packed projection and expert block states this one value, so one
/// launch site compiles one kernel.
pub const ENCODE_CHUNK: u32 = 1024;

/// The encoder's chunk is a whole number of scale blocks; a chunk that were not would have a
/// partial block with no scale to name it.
const _: () = assert!(ENCODE_CHUNK.is_multiple_of(BLOCK_ELEMENTS));

/// The eight magnitudes an E2M1 code states, then the same negated: a code's fourth bit is its
/// sign.
const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// The largest byte that is a finite E4M3 value, 448; the byte after it is the format's NaN.
const E4M3_LARGEST: u8 = 0x7e;

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

/// The NVFP4 encoding of `x`: `(values, scales)`, the packed E2M1 codes and their E4M3 block
/// scales under `global_scale` (module formula).
///
/// `x` is a BF16 or FP16 activation of `[..., k]`, `k` whole sixteen-element blocks, read as its
/// rows; `global_scale` is the `[1]` FP32 encoder scale, the reciprocal of the activation's
/// tensor scale. `values` is `[rows, k / 2]` U8, the even element of each pair in the low
/// nibble; `scales` holds one U8 byte per block, the E4M3 scale's bits, arranged in `layout`
/// ([`ScaleLayout::shape`]) with the padding of the swizzled table zero. Each row's factor is
/// `1 / global_scale`, which a product multiplies its block scales by.
///
/// On CPU tensors a negative element keeps its sign bit when its magnitude rounds to zero, as
/// the device's conversion of a negative zero does; both decode to zero.
///
/// # Errors
///
/// When `x` is not BF16 or FP16 rows of whole blocks, `global_scale` is not one FP32 number, or
/// the operator refuses its arguments.
pub fn nvfp4_quantize(
    x: &Tensor,
    global_scale: &Tensor,
    layout: ScaleLayout,
) -> Result<(Tensor, Tensor)> {
    const OWNER: &str = "nvfp4_quantize";
    let refuse = |problem: String| Error::shape(OWNER, problem);

    let k = columns(OWNER, x)?;
    if !k.is_multiple_of(BLOCK_ELEMENTS) {
        return Err(refuse(format!(
            "rows of {k} elements are not whole sixteen-element blocks"
        )));
    }
    if !matches!(x.dtype(), DType::BF16 | DType::F16) {
        return Err(refuse(format!(
            "{} activations; NVFP4 encodes BF16 or FP16",
            ops::abi::name(x.dtype())
        )));
    }
    if global_scale.dtype() != DType::F32 || global_scale.elem_count() != 1 {
        return Err(refuse(format!(
            "the global scale is {} {:?}, and the encoder reads one FP32 number",
            ops::abi::name(global_scale.dtype()),
            global_scale.dims()
        )));
    }

    let rows = as_rows(OWNER, x)?;
    let count = rows.dims()[0];
    let (packed, blocks) = ((k / 2) as usize, (k / BLOCK_ELEMENTS) as usize);
    let table = layout.shape(count as u64, u64::from(k));
    let table = (table[0] as usize, table[1] as usize);

    if on_cpu(x) {
        let (codes, scale) = encoded(&rows, global_scale)?;

        // Each element's nibble: the index of its magnitude among the eight, and its sign bit.
        let magnitude = codes.abs()?.reshape((count, k as usize))?;
        let negative = rows.to_dtype(ACCUMULATE)?.lt(0.0)?;
        let mut nibble = negative.to_dtype(ACCUMULATE)?.affine(8.0, 0.0)?;
        for step in &E2M1[1..8] {
            nibble = (nibble + magnitude.ge(f64::from(*step))?.to_dtype(ACCUMULATE)?)?;
        }

        // Two nibbles a byte, the even element low.
        let pairs = nibble.reshape((count, packed, 2))?;
        let values = (pairs.narrow(D::Minus1, 0, 1)?
            + pairs.narrow(D::Minus1, 1, 1)?.affine(16.0, 0.0)?)?
        .reshape((count, packed))?
        .to_dtype(DType::U8)?;

        // A scale's byte is its rank among the format's positive values, which ascend with
        // their bits.
        let positive: Vec<f32> = (1..=E4M3_LARGEST)
            .map(|byte| F8E4M3::from_bits(byte).to_f32())
            .collect();
        let bytes = scale
            .unsqueeze(D::Minus1)?
            .broadcast_ge(&Tensor::new(positive.as_slice(), &Device::Cpu)?)?
            .to_dtype(ACCUMULATE)?
            .sum(D::Minus1)?
            .to_dtype(DType::U8)?;

        let scales = match layout {
            ScaleLayout::Linear => bytes,
            ScaleLayout::Swizzled128X4 => {
                Tensor::zeros(table.0 * table.1, DType::U8, &Device::Cpu)?
                    .scatter(
                        &swizzled_offsets(count, blocks, table.1)?,
                        &bytes.flatten_all()?,
                        0,
                    )?
                    .reshape(table)?
            }
        };
        return Ok((values, scales));
    }

    let values = ops::empty((count, packed), DType::U8)?;
    let scales = ops::empty(table, DType::U8)?;
    let factor = ops::empty(count, DType::F32)?;

    launch_quantize(
        &rows,
        global_scale,
        layout,
        (
            ops::operand(&values)?,
            ops::operand(&scales)?,
            ops::operand(&factor)?,
        ),
    )?;
    Ok((values, scales))
}

/// The encoder's one launch: `rows`, `[m, k]`, encoded under `global_scale` into `encoded`, the
/// kernel's three results `(values, scales, factor)`: `[m, k / 2]` packed codes, the block
/// scales of `m` rows in `layout` as bytes, and `[m]` FP32 row factors.
///
/// The results are the caller's: a packed product takes them at its row class, a routed product
/// keeps the row factors beside the codes, and an expert exchange lends the rows its peers read.
pub(super) fn launch_quantize(
    rows: &Tensor,
    global_scale: &Tensor,
    layout: ScaleLayout,
    encoded: (ops::Operand, ops::Operand, ops::Operand),
) -> Result<()> {
    let k = columns("nvfp4_quantize", rows)?;
    let (values, scales, token_scale) = encoded;

    ops::launch::<Nvfp4Quantize>(
        &QuantizeDescriptor {
            input: rows.dtype(),
            columns: k,
            layout,
            block: encode_block(k, ENCODE_CHUNK),
            pdl: PDL,
        },
        QuantizeArgs {
            input: ops::operand(rows)?,
            values,
            scales,
            global_scale: ops::operand(global_scale)?,
            token_scale,
        },
    )?;
    Ok(())
}

/// The encoding of `rows`, `[m, k]`, as FP32 values: `(codes, scale)`, each element's E2M1 value
/// as `[m, k / 16, 16]` and each block's E4M3 scale as `[m, k / 16]`.
///
/// The decoded activation is `codes · scale · (1 / global_scale)`.
pub(super) fn encoded(rows: &Tensor, global_scale: &Tensor) -> Result<(Tensor, Tensor)> {
    let (m, k) = (rows.dims()[0], rows.dims()[1]);
    let blocks = k / BLOCK_ELEMENTS as usize;
    let global = global_scale.reshape(())?;

    // Each block's E4M3 scale under the global scale.
    let x = rows.to_dtype(ACCUMULATE)?.reshape((m, blocks, 16))?;
    let amax = x.abs()?.max(D::Minus1)?;
    let scale = (amax / 6.0)?
        .broadcast_mul(&global)?
        .to_dtype(DType::F8E4M3)?
        .to_dtype(ACCUMULATE)?;

    // A block whose scale rounds to zero encodes to zeros: the encoder multiplies by zero
    // there, where the reciprocal of a zero scale is not a number.
    let relative = scale.broadcast_mul(&global.recip()?)?;
    let inverse = relative
        .ne(0.0)?
        .where_cond(&relative.recip()?, &relative.zeros_like()?)?;
    let codes = e2m1(&x.broadcast_mul(&inverse.unsqueeze(D::Minus1)?)?)?;

    Ok((codes, scale))
}

/// The FP32 matrices a packed field states, before any tensor scale: `values` is
/// `[rows, k / 2]` U8, or a stack `[experts, rows, k / 2]` of them, two E2M1 codes a byte with
/// the even element in the low nibble; `scales` its E4M3 block scales in `layout`, one table per
/// matrix. The result is `code · block scale`, `[rows, k]` or `[experts, rows, k]`.
pub(super) fn decoded(values: &Tensor, scales: &Tensor, layout: ScaleLayout) -> Result<Tensor> {
    let (experts, rows, packed) = match *values.dims() {
        [rows, packed] => (1, rows, packed),
        [experts, rows, packed] => (experts, rows, packed),
        _ => {
            return Err(Error::shape(
                "nvfp4",
                format!("a packed field of {:?}", values.dims()),
            ));
        }
    };
    let (k, blocks) = (packed * 2, packed * 2 / BLOCK_ELEMENTS as usize);

    let high = values.broadcast_div(&integer(16, values)?)?;
    let low = values.broadcast_remainder(&integer(16, values)?)?;
    let nibbles = Tensor::stack(&[low, high], D::Minus1)?.flatten_all()?;
    let codes = Tensor::new(&E2M1, &Device::Cpu)?
        .index_select(&nibbles, 0)?
        .reshape((experts, rows, blocks, 16))?;

    // Each matrix's table is read at the offsets its layout places its `[rows, blocks]` scales.
    let tables = scales.reshape((experts, scales.elem_count() / experts))?;
    let placed = match layout {
        ScaleLayout::Linear => tables,
        ScaleLayout::Swizzled128X4 => {
            let padded_columns = scales.dims().last().copied().unwrap_or(0);
            tables.index_select(&swizzled_offsets(rows, blocks, padded_columns)?, 1)?
        }
    };
    let block_scales = e4m3_values(&placed)?.reshape((experts, rows, blocks, 1))?;

    let mut dims = values.dims().to_vec();
    if let Some(last) = dims.last_mut() {
        *last = k;
    }
    Ok(codes.broadcast_mul(&block_scales)?.reshape(dims)?)
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

    // Raw bytes are the elements' bits: read as E4M3, they are cast like a stored scale.
    Ok(scales.view(DType::F8E4M3)?.to_dtype(ACCUMULATE)?)
}

/// `value` as a one-element tensor of `like`'s integer element type, for integer arithmetic,
/// which the scalar operators of the tensor library evaluate in floating point.
pub(super) fn integer(value: u32, like: &Tensor) -> Result<Tensor> {
    Ok(Tensor::new(value, &Device::Cpu)?.to_dtype(like.dtype())?)
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

    let within = times(row.broadcast_remainder(&integer(32, &row)?)?, 16)?;
    let group = times(
        over(&row.broadcast_remainder(&integer(128, &row)?)?, 32)?,
        4,
    )?;
    let tile = times(over(&row, 128)?, 128 * padded_columns)?;
    let row_offset = ((within + group)? + tile)?;

    let column_offset =
        (column.broadcast_remainder(&integer(4, &column)?)? + times(over(&column, 4)?, 512)?)?;

    Ok(row_offset
        .unsqueeze(1)?
        .broadcast_add(&column_offset.unsqueeze(0)?)?
        .flatten_all()?)
}

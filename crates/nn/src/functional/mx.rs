//! The MX formats' dense operators: the MXFP8 encoding of an activation and the product of an
//! MXFP8 activation and weight ([`ops::mx`]).
//!
//! ```text
//! mxfp8_quantize          x                       -> E4M3 codes and one UE8M0 scale per 32
//!                                                    elements
//! linear_mxfp8            x, MXFP8 weight (+ bias) -> x̂ · Ŵᵀ in FP32, rounded once (+ bias)
//! grouped_linear_mxfp8    x, MXFP8 weight, groups  -> each group's x̂_g · Ŵ_gᵀ, side by side
//! ```
//!
//! The CPU formulas compute the encoding element for element as `ops::mx` states it, and the
//! product over the decoded values in FP32: they are the references the kernels are held to.

use ops::mx::{
    BLOCK, GemmArgs, GemmDescriptor, Mxfp8Gemm, Mxfp8Quantize, QuantizeArgs, QuantizeDescriptor,
    ScaleLayout, ScaleRule, block_exponent, ue8m0,
};
use tensor::{DType, Device, Tensor};

use super::{ACCUMULATE, Fused, as_rows, columns, compute, like_rows, vector};
use crate::error::{Error, Result};
use crate::quantization::Mxfp8Weight;

/// The largest finite E4M3 magnitude.
const E4M3_MAX: f32 = ops::fp8::E4M3_MAX;

/// The MXFP8 encoding of `x` `[..., k]`: `(codes, scales)`, its E4M3 codes `[rows, k]` and its
/// UE8M0 scales over `[rows, k]` in `layout` ([`ScaleLayout::Linear`] or
/// [`ScaleLayout::PackedMnMajor`]), as `ops::mx` states the encoding:
///
/// ```text
/// scale = 2^⌈log₂ fl(max(amax, 10⁻⁴) · fl(1/448))⌉     per 32 elements of a row
/// code  = e4m3(clamp(x / scale, ±448))                 round to nearest even
/// ```
///
/// `x` is BF16 or FP16 and `k` whole packed scale words of 128 elements.
///
/// # Errors
///
/// When `x` is of another type or width, or the operator refuses its arguments.
pub fn mxfp8_quantize(x: &Tensor, layout: ScaleLayout) -> Result<(Tensor, Tensor)> {
    const OWNER: &str = QuantizeCall::OWNER;

    let k = columns(OWNER, x)?;
    if !matches!(x.dtype(), DType::BF16 | DType::F16) || !k.is_multiple_of(128) {
        return Err(Error::shape(
            OWNER,
            format!(
                "MXFP8 encodes BF16 or FP16 rows of whole 128-element words, not {} {:?}",
                ops::abi::name(x.dtype()),
                x.dims()
            ),
        ));
    }
    let rows = as_rows(OWNER, x)?;

    compute(QuantizeCall { rows, layout, k })
}

/// A checked call of [`mxfp8_quantize`].
struct QuantizeCall {
    rows: Tensor,
    layout: ScaleLayout,
    k: u32,
}

impl QuantizeCall {
    const OWNER: &'static str = "mxfp8_quantize";
}

impl Fused for QuantizeCall {
    type Output = (Tensor, Tensor);

    fn device(&self) -> &Device {
        self.rows.device()
    }

    fn cpu(self) -> Result<Self::Output> {
        let Self { rows, layout, k } = self;
        let (count, k) = (rows.dims()[0], k as usize);
        let (quotients, scales) = encoded_rows(&rows, ScaleRule::Covering)?;

        let codes =
            Tensor::from_vec(quotients, (count, k), &Device::Cpu)?.to_dtype(DType::F8E4M3)?;
        let table = ops::mx::arranged(&scales, count as u64, k as u64, layout);
        let shape = layout.shape(count as u64, k as u64);
        let shape = (shape[0] as usize, shape[1] as usize);
        let scales =
            Tensor::from_raw_buffer(&table, layout.dtype(), &[shape.0, shape.1], &Device::Cpu)?;
        Ok((codes, scales))
    }

    fn launch(self) -> Result<Self::Output> {
        let Self { rows, layout, k } = self;
        let count = rows.dims()[0];
        let codes = ops::empty((count, k as usize), DType::F8E4M3)?;
        let shape = layout.shape(count as u64, u64::from(k));
        let scales = ops::empty((shape[0] as usize, shape[1] as usize), layout.dtype())?;

        ops::launch::<Mxfp8Quantize>(
            &QuantizeDescriptor::new(rows.dtype(), k, layout),
            QuantizeArgs {
                input: ops::operand(&rows)?,
                codes: ops::operand(&codes)?,
                scales: ops::operand(&scales)?,
            },
        )?;
        Ok((codes, scales))
    }
}

/// The encoding of a CPU tensor's `[rows, k]` rows under the scales `rule` chooses: each
/// element's quotient by its block's scale, clamped to ±448 (the value its E4M3 code rounds), and
/// each block's UE8M0 byte, the linear `[rows, k / 32]` table.
pub(super) fn encoded_rows(rows: &Tensor, rule: ScaleRule) -> Result<(Vec<f32>, Vec<u8>)> {
    let values = rows.to_dtype(ACCUMULATE)?.flatten_all()?.to_vec1::<f32>()?;
    let block = BLOCK as usize;

    let mut quotients = Vec::with_capacity(values.len());
    let mut scales = Vec::with_capacity(values.len() / block);
    for chunk in values.chunks(block) {
        let amax = chunk
            .iter()
            .fold(0.0_f32, |amax, value| amax.max(value.abs()));
        let byte = block_exponent(amax, rule);
        let scale = ue8m0(byte);
        scales.push(byte);
        quotients.extend(
            chunk
                .iter()
                .map(|value| (value / scale).clamp(-E4M3_MAX, E4M3_MAX)),
        );
    }
    Ok((quotients, scales))
}

/// The FP32 values an MXFP8 matrix's E4M3 `codes` `[rows, k]` stand for under its `scales`
/// in `layout`: each code times its 32-element block's power of two.
pub(super) fn decoded_mxfp8(
    codes: &Tensor,
    scales: &Tensor,
    layout: ScaleLayout,
) -> Result<Tensor> {
    let &[rows, k] = codes.dims() else {
        return Err(Error::shape(
            "mxfp8",
            format!("a matrix of codes, not {:?}", codes.dims()),
        ));
    };
    let linear = ops::mx::linear(&table_bytes(scales)?, rows as u64, k as u64, layout);
    let factors: Vec<f32> = linear
        .iter()
        .flat_map(|byte| std::iter::repeat_n(ue8m0(*byte), BLOCK as usize))
        .collect();
    let factors = Tensor::from_vec(factors, (rows, k), &Device::Cpu)?;
    Ok((codes.to_dtype(ACCUMULATE)? * factors)?)
}

/// The magnitudes an E2M1 code states; a code's fourth bit is its sign.
const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// The FP32 values an MXFP4 stack's packed codes `[experts, rows, k / 2]` stand for under its
/// scales in `layout`, `[experts, rows, k]`: each code (the even element in the low nibble) times
/// its 32-element block's power of two. The rows are the stack's stored rows, in whatever
/// arrangement it holds them.
pub(super) fn decoded_mxfp4(
    codes: &Tensor,
    scales: &Tensor,
    layout: ScaleLayout,
) -> Result<Tensor> {
    let &[experts, rows, packed] = codes.dims() else {
        return Err(Error::shape(
            "mxfp4",
            format!("a stack of packed codes, not {:?}", codes.dims()),
        ));
    };
    let (k, block) = (packed * 2, BLOCK as usize);
    let bytes = codes.flatten_all()?.to_vec1::<u8>()?;
    let table = table_bytes(scales)?;
    let entry = table.len() / experts.max(1);

    let mut values = Vec::with_capacity(experts * rows * k);
    for expert in 0..experts {
        let linear = ops::mx::linear(
            &table[expert * entry..(expert + 1) * entry],
            rows as u64,
            k as u64,
            layout,
        );
        let matrix = &bytes[expert * rows * packed..(expert + 1) * rows * packed];
        for row in 0..rows {
            for element in 0..k {
                let byte = matrix[row * packed + element / 2];
                let code = if element % 2 == 0 {
                    byte & 0x0f
                } else {
                    byte >> 4
                };
                let sign = if code & 0x08 == 0 { 1.0 } else { -1.0 };
                let scale = ue8m0(linear[row * (k / block) + element / block]);
                values.push(sign * E2M1[usize::from(code & 0x07)] * scale);
            }
        }
    }
    Ok(Tensor::from_vec(values, (experts, rows, k), &Device::Cpu)?)
}

/// The bytes of a CPU scale table: its bytes, or its words' little-endian bytes.
pub(super) fn table_bytes(scales: &Tensor) -> Result<Vec<u8>> {
    let flat = scales.flatten_all()?;
    Ok(match flat.dtype() {
        DType::U8 => flat.to_vec1::<u8>()?,
        DType::U32 => flat
            .to_vec1::<u32>()?
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect(),
        other => {
            return Err(Error::shape(
                "mxfp8",
                format!("a scale table of {}", ops::abi::name(other)),
            ));
        }
    })
}

/// `x · Wᵀ + bias` over an MXFP8 weight ([`Mxfp8Weight`]): the activation encoded per call and
/// the product over the two encodings,
///
/// ```text
/// x̂   = mxfp8_quantize(x)
/// out = Σ_k x̂[·, k] · Ŵ[·, k]          over the decoded values, in FP32, rounded once
///     (+ bias)                          added to the rounded product, rounded again
/// ```
///
/// `x` is `[..., k]` BF16; the weight `[n, k]` E4M3 with its scales in
/// [`GemmDescriptor::SCALE_LAYOUT`]. The result is `[..., n]` BF16.
///
/// # Errors
///
/// When the weight is not an MXFP8 matrix over `x`'s rows in the product's layout, the bias is
/// another width, or an operator refuses its arguments.
pub fn linear_mxfp8(x: &Tensor, weight: &Mxfp8Weight, bias: Option<&Tensor>) -> Result<Tensor> {
    product(LinearCall::OWNER, x, weight, 1, bias)
}

/// The grouped product of `x` and an MXFP8 weight of `groups` groups, in one launch: group `g`
/// multiplies its block of `k` columns of the encoded activation with its block of `n` rows of
/// the weight into its block of `n` output columns,
///
/// ```text
/// x̂            = mxfp8_quantize(x)                  the whole row, `groups · k` elements
/// out[·, g n + j] = Σ_i x̂[·, g k + i] · Ŵ[g n + j, i]   in FP32, rounded once
/// ```
///
/// which is the projection of each group's slice by its own `[n, k]` weight, the outputs
/// concatenated (DeepSeek-V4.1's `wo_a`). `x` is `[..., groups · k]` BF16; the weight
/// `[groups · n, k]` E4M3 with its scales in [`GemmDescriptor::SCALE_LAYOUT`] as the table of the
/// `[n, groups · k]` matrix whose column block `g` is group `g`'s rows
/// ([`crate::weights::WeightLoader::mxfp8_groups`]). The result is `[..., groups · n]` BF16.
///
/// # Errors
///
/// When the weight is not an MXFP8 matrix of `groups` groups over `x`'s rows in the product's
/// layout, or an operator refuses its arguments.
pub fn grouped_linear_mxfp8(x: &Tensor, weight: &Mxfp8Weight, groups: u32) -> Result<Tensor> {
    product("grouped_linear_mxfp8", x, weight, groups, None)
}

/// [`linear_mxfp8`] and [`grouped_linear_mxfp8`]: `groups` products over the encoding of `x`,
/// checked for `owner`.
fn product(
    owner: &'static str,
    x: &Tensor,
    weight: &Mxfp8Weight,
    groups: u32,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    let width = columns(owner, x)?;
    let &[rows, contraction] = weight.codes.dims() else {
        return Err(Error::shape(
            owner,
            format!("a weight is a matrix, not {:?}", weight.codes.dims()),
        ));
    };
    let k = width / groups.max(1);
    let n = rows / groups.max(1) as usize;
    let scales = GemmDescriptor::SCALE_LAYOUT.shape(n as u64, u64::from(width));
    if groups == 0
        || width != groups * k
        || rows != groups as usize * n
        || contraction != k as usize
        || weight.scales.dims() != [scales[0] as usize, scales[1] as usize]
        || weight.codes.dtype() != DType::F8E4M3
        || weight.layout != GemmDescriptor::SCALE_LAYOUT
        || x.dtype() != DType::BF16
    {
        return Err(Error::shape(
            owner,
            format!(
                "{groups} MXFP8 products of BF16 rows of {groups} x k over a [{groups} x n, k] \
                 E4M3 weight with {} scales of [n, {groups} x k], not {} rows of {width} and a \
                 {} {:?} weight with {:?} {:?} scales",
                GemmDescriptor::SCALE_LAYOUT,
                ops::abi::name(x.dtype()),
                ops::abi::name(weight.codes.dtype()),
                weight.codes.dims(),
                weight.scales.dims(),
                weight.layout
            ),
        ));
    }
    let n = u32::try_from(n).map_err(|_| Error::shape(owner, format!("{n} output channels")))?;
    if let Some(bias) = bias {
        vector(owner, "bias", bias, groups * n)?;
    }
    let rows = as_rows(owner, x)?;

    compute(LinearCall {
        x,
        rows,
        weight,
        bias,
        n,
        groups,
    })
}

/// A checked call of [`linear_mxfp8`] or [`grouped_linear_mxfp8`].
struct LinearCall<'t> {
    x: &'t Tensor,
    rows: Tensor,
    weight: &'t Mxfp8Weight,
    bias: Option<&'t Tensor>,
    /// Output channels of each group.
    n: u32,
    groups: u32,
}

impl LinearCall<'_> {
    const OWNER: &'static str = "linear_mxfp8";
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
            n,
            groups,
        } = self;
        let (count, width) = (rows.dims()[0], rows.dims()[1]);
        let (groups, n) = (groups as usize, n as usize);
        let k = width / groups;
        let (quotients, scales) = encoded_rows(&rows, ScaleRule::Covering)?;
        let codes =
            Tensor::from_vec(quotients, (count, width), &Device::Cpu)?.to_dtype(DType::F8E4M3)?;
        let scales = Tensor::from_vec(scales, count * width / BLOCK as usize, &Device::Cpu)?;
        let activation = decoded_mxfp8(&codes, &scales, ScaleLayout::Linear)?;

        // The weight's groups side by side along the contraction, as its scales are laid out:
        // `[n, groups · k]`, column block `g` group `g`'s rows.
        let beside = weight
            .codes
            .reshape((groups, n, k))?
            .permute((1, 0, 2))?
            .reshape((n, groups * k))?
            .contiguous()?;
        let matrix = decoded_mxfp8(&beside, &weight.scales, weight.layout)?;
        let products = (0..groups)
            .map(|group| {
                let columns = |values: &Tensor| values.narrow(1, group * k, k);
                Ok(columns(&activation)?.matmul(&columns(&matrix)?.t()?)?)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut out = Tensor::cat(&products, 1)?.to_dtype(x.dtype())?;
        if let Some(bias) = bias {
            out = out.broadcast_add(&bias.to_dtype(x.dtype())?)?;
        }
        like_rows(out, x)
    }

    fn launch(self) -> Result<Self::Output> {
        let Self {
            x,
            rows,
            weight,
            bias,
            n,
            groups,
        } = self;
        let (count, width) = (rows.dims()[0], rows.dims()[1] as u32);

        // 1. The activation's codes and its packed scale words, the groups' blocks of one row.
        let (codes, scales) = mxfp8_quantize(&rows, GemmDescriptor::SCALE_LAYOUT)?;

        // 2. Every group's product over the two encodings in one launch, its tile chosen for the
        //    rows' class.
        let out = ops::empty((count, (groups * n) as usize), DType::BF16)?;
        ops::launch::<Mxfp8Gemm>(
            &GemmDescriptor::new(count as u64, n, width / groups, groups),
            GemmArgs {
                input: ops::operand(&codes)?,
                input_scales: ops::operand(&scales)?,
                weight: ops::operand(&weight.codes)?,
                weight_scales: ops::operand(&weight.scales)?,
                out: ops::operand(&out)?,
            },
        )?;

        let out = match bias {
            Some(bias) => out.broadcast_add(bias)?,
            None => out,
        };
        like_rows(out, x)
    }
}

//! Hashed n-gram memory: the table rows the n-grams ending at each token name, and the gate that
//! adds what they recall into the streams of a residual (`ops::ngram`).
//!
//! ```text
//! ngram_hash   row[(i − 1)·H + h] = (⊕_{b ≤ i} normalize[t_b] · μ_b) mod P[i − 1, h] + O[i − 1, h]
//! ngram_gate   h'_k = round(h_k + sigmoid(copysign(√max(|dot_k|, 10⁻⁶), dot_k)) · value)
//! ```
//!
//! with `t_b` the token `b` back (its own at `b = 0`) and `dot_k` the agreement of stream `k`
//! with its key (`ops::ngram`). The rows themselves are an MXFP8 table's
//! ([`embedding_mxfp8`](super::embedding_mxfp8)).

use ops::ngram::{GateArgs, GateDescriptor, HashArgs, HashDescriptor, NgramGate, NgramHash};
use tensor::{D, DType, Device, Tensor};

use super::{ACCUMULATE, Fused, compute};
use crate::error::{Error, Result};

/// The agreement's floor before its square root: a small agreement's gate stays near one half
/// instead of following an unbounded slope.
const GATE_FLOOR: f64 = 1e-6;

/// The table rows of every n-gram that ends at each token: `[tokens, (N − 1) · H]` I64.
///
/// `context` is `[tokens, N]` U32 or I32: each token's `N − 1` vocabulary ids before it, oldest
/// first, then its own, a position before the request's first token holding the id the model
/// fills a context with. `normalize` is the `[vocabulary]` I32 map onto the normalized ids,
/// `multipliers` the `[N]` I64 multipliers (the token's own first), and `moduli` and `offsets`
/// the `[N − 1, H]` I64 prime moduli and first rows of each head of each n-gram length.
///
/// # Errors
///
/// When an operand has other extents or types, or the operator refuses its arguments.
pub fn ngram_hash(
    context: &Tensor,
    normalize: &Tensor,
    multipliers: &Tensor,
    moduli: &Tensor,
    offsets: &Tensor,
) -> Result<Tensor> {
    const OWNER: &str = HashCall::OWNER;

    let &[_, order] = context.dims() else {
        return Err(Error::shape(
            OWNER,
            format!("a context of {:?} is not [tokens, order]", context.dims()),
        ));
    };
    let heads = moduli.dims().get(1).copied().unwrap_or(0);
    let wrong = |what: &str, tensor: &Tensor, dims: &[usize], dtype: DType| {
        (tensor.dims() != dims || tensor.dtype() != dtype).then(|| {
            Error::shape(
                OWNER,
                format!(
                    "the {what} are {} {:?}, {} {dims:?} expected",
                    ops::abi::name(tensor.dtype()),
                    tensor.dims(),
                    ops::abi::name(dtype)
                ),
            )
        })
    };
    if order < 2 || heads == 0 || !matches!(context.dtype(), DType::U32 | DType::I32) {
        return Err(Error::shape(
            OWNER,
            format!(
                "a context of {} {:?} and moduli of {:?}: U32 or I32 contexts of at least a bigram \
                 and one head",
                ops::abi::name(context.dtype()),
                context.dims(),
                moduli.dims()
            ),
        ));
    }
    if normalize.rank() != 1 || normalize.elem_count() == 0 || normalize.dtype() != DType::I32 {
        return Err(Error::shape(
            OWNER,
            format!("a normalization table of {:?}", normalize.dims()),
        ));
    }
    if let Some(error) = [
        wrong("multipliers", multipliers, &[order], DType::I64),
        wrong("moduli", moduli, &[order - 1, heads], DType::I64),
        wrong("offsets", offsets, &[order - 1, heads], DType::I64),
    ]
    .into_iter()
    .flatten()
    .next()
    {
        return Err(error);
    }

    compute(HashCall {
        context,
        normalize,
        multipliers,
        moduli,
        offsets,
    })
}

/// A checked call of [`ngram_hash`].
struct HashCall<'t> {
    context: &'t Tensor,
    normalize: &'t Tensor,
    multipliers: &'t Tensor,
    moduli: &'t Tensor,
    offsets: &'t Tensor,
}

impl HashCall<'_> {
    const OWNER: &'static str = "ngram_hash";
}

impl Fused for HashCall<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.context.device()
    }

    fn cpu(self) -> Result<Tensor> {
        let Self {
            context,
            normalize,
            multipliers,
            moduli,
            offsets,
        } = self;
        let (tokens, order) = context.dims2()?;
        let vocabulary = normalize.elem_count();

        // Each token `back` tokens before, normalized; an id past the table reads its last entry.
        let ids = context
            .to_dtype(DType::I64)?
            .clamp(0.0, (vocabulary - 1) as f64)?
            .to_dtype(DType::U32)?
            .flatten_all()?;
        let normalized = normalize
            .index_select(&ids, 0)?
            .to_dtype(DType::I64)?
            .reshape((tokens, order))?;

        let mut hashes = Vec::with_capacity(order - 1);
        let mut rolling: Option<Tensor> = None;
        for back in 0..order {
            let term = normalized
                .narrow(1, order - 1 - back, 1)?
                .broadcast_mul(&multipliers.narrow(0, back, 1)?)?;
            let next = match rolling {
                None => term,
                Some(rolling) => {
                    let mixed = rolling.broadcast_bitwise_xor(&term)?;
                    let modulus = moduli.narrow(0, back - 1, 1)?;
                    let offset = offsets.narrow(0, back - 1, 1)?;
                    // The exclusive-or of non-negative values is non-negative, so the remainder
                    // is the non-negative one.
                    hashes.push(
                        mixed
                            .broadcast_remainder(&modulus)?
                            .broadcast_add(&offset)?,
                    );
                    mixed
                }
            };
            rolling = Some(next);
        }
        Ok(Tensor::cat(&hashes, 1)?)
    }

    fn launch(self) -> Result<Tensor> {
        let Self {
            context,
            normalize,
            multipliers,
            moduli,
            offsets,
        } = self;
        let (tokens, order) = context.dims2()?;
        let heads = moduli.dim(1)?;
        let out = ops::empty((tokens, (order - 1) * heads), DType::I64)?;
        if tokens == 0 {
            return Ok(out);
        }

        let wide = |value: usize| {
            u32::try_from(value)
                .map_err(|_| Error::shape(Self::OWNER, format!("an extent of {value}")))
        };
        ops::launch::<NgramHash>(
            &HashDescriptor::new(context.dtype(), wide(order)?, wide(heads)?),
            HashArgs {
                context: ops::operand(&context.contiguous()?)?,
                normalize: ops::operand(normalize)?,
                multipliers: ops::operand(multipliers)?,
                moduli: ops::operand(&moduli.contiguous()?)?,
                offsets: ops::operand(&offsets.contiguous()?)?,
                out: ops::operand(&out)?,
            },
        )?;
        Ok(out)
    }
}

/// Each of `streams`' `[tokens, n, d]` streams plus the recalled value under the gate of its
/// key's agreement with the stream: `[tokens, n, d]` in the streams' type.
///
/// `projected` is `[tokens, (n + 1) · d]` in the streams' type, each token's key of every stream
/// and then its value (rows of unit stride, which may sit in a wider buffer); `query` and `key`
/// are the gate's `[n, d]` weights, read only as their FP32 product. `eps` is added to both mean
/// squares.
///
/// # Errors
///
/// When an operand has other extents or types, or the operator refuses its arguments.
pub fn ngram_gate(
    streams: &Tensor,
    projected: &Tensor,
    query: &Tensor,
    key: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    const OWNER: &str = GateCall::OWNER;

    let &[tokens, count, width] = streams.dims() else {
        return Err(Error::shape(
            OWNER,
            format!(
                "streams of {:?} are not [tokens, streams, hidden]",
                streams.dims()
            ),
        ));
    };
    let dtype = streams.dtype();
    for (name, tensor, dims) in [
        (
            "projected rows",
            projected,
            [tokens, (count + 1) * width].as_slice(),
        ),
        ("query weight", query, &[count, width]),
        ("key weight", key, &[count, width]),
    ] {
        if tensor.dims() != dims || tensor.dtype() != dtype {
            return Err(Error::shape(
                OWNER,
                format!(
                    "the {name} are {} {:?}, {} {dims:?} expected",
                    ops::abi::name(tensor.dtype()),
                    tensor.dims(),
                    ops::abi::name(dtype)
                ),
            ));
        }
    }

    compute(GateCall {
        streams,
        projected,
        query,
        key,
        eps,
    })
}

/// A checked call of [`ngram_gate`].
struct GateCall<'t> {
    streams: &'t Tensor,
    projected: &'t Tensor,
    query: &'t Tensor,
    key: &'t Tensor,
    eps: f32,
}

impl GateCall<'_> {
    const OWNER: &'static str = "ngram_gate";
}

/// `1/√width` as FP32, as the reference's `width ** -0.5` reaches its FP32 product.
fn agreement_scale(width: usize) -> f32 {
    (width as f64).powf(-0.5) as f32
}

impl Fused for GateCall<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.streams.device()
    }

    fn cpu(self) -> Result<Tensor> {
        let Self {
            streams,
            projected,
            query,
            key,
            eps,
        } = self;
        let (tokens, count, width) = streams.dims3()?;
        let h = streams.to_dtype(ACCUMULATE)?;
        let projected = projected.to_dtype(ACCUMULATE)?;
        let keys = projected
            .narrow(1, 0, count * width)?
            .reshape((tokens, count, width))?;
        let value = projected.narrow(1, count * width, width)?.unsqueeze(1)?;
        let weight = query
            .to_dtype(ACCUMULATE)?
            .mul(&key.to_dtype(ACCUMULATE)?)?
            .unsqueeze(0)?;

        let inverse = |rows: &Tensor| -> Result<Tensor> {
            Ok((rows.sqr()?.mean_keepdim(D::Minus1)? + f64::from(eps))?
                .sqrt()?
                .recip()?)
        };
        let scale = inverse(&h)?.mul(&inverse(&keys)?)?;
        let dot = h
            .broadcast_mul(&weight)?
            .mul(&keys)?
            .sum_keepdim(D::Minus1)?
            .mul(&scale)?
            .affine(f64::from(agreement_scale(width)), 0.0)?;

        // The root takes the agreement's sign bit, as `copysign` does.
        let root = dot.abs()?.maximum(GATE_FLOOR)?.sqrt()?;
        let negative = dot
            .contiguous()?
            .view(DType::U32)?
            .ge(f64::from(1_u32 << 31))?;
        let gate = negative.where_cond(&root.neg()?, &root)?.sigmoid()?;

        Ok((h + gate.broadcast_mul(&value)?)?.to_dtype(streams.dtype())?)
    }

    fn launch(self) -> Result<Tensor> {
        let Self {
            streams,
            projected,
            query,
            key,
            eps,
        } = self;
        let (tokens, count, width) = streams.dims3()?;
        let out = ops::empty(streams.dims(), streams.dtype())?;
        if tokens == 0 {
            return Ok(out);
        }

        let extent = |value: usize| {
            u32::try_from(value)
                .map_err(|_| Error::shape(Self::OWNER, format!("an extent of {value}")))
        };
        let projected = if projected.stride()[1] == 1 {
            projected.clone()
        } else {
            projected.contiguous()?
        };
        ops::launch::<NgramGate>(
            &GateDescriptor::new(streams.dtype(), extent(count)?, extent(width)?),
            GateArgs {
                streams: ops::operand(&streams.contiguous()?)?,
                projected: ops::operand(&projected)?,
                query: ops::operand(&query.contiguous()?)?,
                key: ops::operand(&key.contiguous()?)?,
                out: ops::operand(&out)?,
                eps,
                scale: agreement_scale(width),
            },
        )?;
        Ok(out)
    }
}

//! Manifold-constrained hyper-connections (mHC): the residual of several streams a sub-layer reads
//! its input from and writes its output back into (`ops::mhc`).
//!
//! The streams are `[tokens, n, d]`. A sub-layer's connection computes its mixes from the streams
//! it reaches, a pre-mix, a post-mix and a doubly stochastic combination matrix
//! ([`MhcMixes`]), and in the shifted form the reference evaluates, the sub-layer's input is
//! read with the pre-mix of the connection before it:
//!
//! ```text
//! mhc_boundary   h ← post(h, y_prev, mixes_prev);  mixes ← mixes(h);  x ← pre(h, pre_carried)
//! mhc_post       h' = round(post_k · y + Σ_j comb[j, k] · h_j)
//! mhc_pre        x  = rms_norm(round(Σ_j pre_j · h_j)) · w
//! ```
//!
//! Each function's CPU formula computes in FP32 and rounds where the kernel rounds: the written
//! streams, the collapsed sum before its normalization, and the normalized input; every product
//! is rounded before the sum it enters, as the reference's separate operations round them.

use ops::mhc::{
    BoundaryArgs, BoundaryDescriptor, MhcBoundary, MhcPost, MhcPre, Pending, PostArgs,
    PostDescriptor, PreArgs, PreDescriptor,
};
use tensor::{D, DType, Device, Tensor};

use super::{ACCUMULATE, Fused, compute};
use crate::error::{Error, Result};

/// The mixes one connection computes from the streams it reaches, FP32 per token.
#[derive(Clone, Debug)]
pub struct MhcMixes {
    /// `[tokens, n]`: the pre-mix, which the next sub-layer's input is read with.
    pub pre: Tensor,
    /// `[tokens, n]`: the post-mix, the gain of the sub-layer's output in each stream.
    pub post: Tensor,
    /// `[tokens, n, n]`: the combination matrix, row `j` the stream read and column `k` the
    /// stream written.
    pub comb: Tensor,
}

/// The mixing parameters of one connection, as a boundary reads them.
#[derive(Clone, Copy, Debug)]
pub struct MhcMixing<'t> {
    /// `[n·(n + 2), n·d]` FP32: the projection of the flattened streams onto the mixes.
    pub projection: &'t Tensor,
    /// `[3]` FP32: the scales of the pre-mix, the post-mix and the combination logits.
    pub scale: &'t Tensor,
    /// `[n·(n + 2)]` FP32: the mixes' biases.
    pub base: &'t Tensor,
    /// The Sinkhorn iterations of the combination matrix.
    pub iterations: u32,
    /// The epsilon of the flattened streams' mean square.
    pub rms_eps: f32,
    /// The epsilon of the pre-mix and of every Sinkhorn step.
    pub eps: f32,
}

/// The streams' count and width, `(n, d)`, of `streams`, `[tokens, n, d]`.
fn extents(owner: &'static str, streams: &Tensor) -> Result<(usize, usize, usize)> {
    match *streams.dims() {
        [tokens, count, width] if count > 0 && width > 0 => Ok((tokens, count, width)),
        _ => Err(Error::shape(
            owner,
            format!(
                "streams of {:?} are not [tokens, streams, hidden]",
                streams.dims()
            ),
        )),
    }
}

/// Checks that `tensor`, the operand `name`, is `dims` of `dtype`.
fn exactly(
    owner: &'static str,
    name: &str,
    tensor: &Tensor,
    dims: &[usize],
    dtype: DType,
) -> Result<()> {
    if tensor.dims() != dims || tensor.dtype() != dtype {
        return Err(Error::shape(
            owner,
            format!(
                "the {name} is {} {:?}, {} {dims:?} expected",
                ops::abi::name(tensor.dtype()),
                tensor.dims(),
                ops::abi::name(dtype)
            ),
        ));
    }
    Ok(())
}

/// Checks a connection's mixes of `tokens` tokens of `count` streams.
fn mixes_of(owner: &'static str, mixes: &MhcMixes, tokens: usize, count: usize) -> Result<()> {
    exactly(owner, "pre-mix", &mixes.pre, &[tokens, count], DType::F32)?;
    exactly(owner, "post-mix", &mixes.post, &[tokens, count], DType::F32)?;
    exactly(
        owner,
        "combination",
        &mixes.comb,
        &[tokens, count, count],
        DType::F32,
    )
}

/// A u32 extent of a descriptor.
fn extent(owner: &'static str, value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::shape(owner, format!("an extent of {value}")))
}

/// One boundary between two sub-layers: `pending`, the previous sub-layer's output with the
/// mixes it was read with, written back into `streams` first; the mixes of the next sub-layer's
/// connection `mixing` computed from the streams it reaches; and that sub-layer's input read from
/// them with `pre`, the pre-mix carried from the connection before, and normalized with the gain
/// `norm`.
///
/// `streams` is `[tokens, n, d]` of a floating type, `pending`'s output `[tokens, d]` of the same
/// type, `pre` `[tokens, n]` FP32 and `norm` `[d]` of the streams' type. Returns the streams the
/// boundary reached (the written ones when an output was pending, `streams` itself otherwise),
/// the input `[tokens, d]` and the connection's mixes.
///
/// # Errors
///
/// When an operand has other extents or types, or the operator refuses its arguments.
pub fn mhc_boundary(
    streams: &Tensor,
    pending: Option<(&Tensor, &MhcMixes)>,
    pre: &Tensor,
    mixing: MhcMixing<'_>,
    norm: &Tensor,
    norm_eps: f32,
) -> Result<(Tensor, Tensor, MhcMixes)> {
    const OWNER: &str = BoundaryCall::OWNER;

    let (tokens, count, width) = extents(OWNER, streams)?;
    let dtype = streams.dtype();
    let mixes = count * (count + 2);
    if let Some((output, previous)) = pending {
        exactly(OWNER, "pending output", output, &[tokens, width], dtype)?;
        mixes_of(OWNER, previous, tokens, count)?;
    }
    exactly(OWNER, "carried pre-mix", pre, &[tokens, count], DType::F32)?;
    exactly(
        OWNER,
        "projection",
        mixing.projection,
        &[mixes, count * width],
        DType::F32,
    )?;
    exactly(OWNER, "scale", mixing.scale, &[3], DType::F32)?;
    exactly(OWNER, "base", mixing.base, &[mixes], DType::F32)?;
    exactly(OWNER, "norm", norm, &[width], dtype)?;
    if mixing.iterations == 0 {
        return Err(Error::shape(
            OWNER,
            "a combination matrix of no Sinkhorn iteration".to_owned(),
        ));
    }

    compute(BoundaryCall {
        streams,
        pending,
        pre,
        mixing,
        norm,
        norm_eps,
    })
}

/// A checked call of [`mhc_boundary`].
struct BoundaryCall<'t> {
    streams: &'t Tensor,
    pending: Option<(&'t Tensor, &'t MhcMixes)>,
    pre: &'t Tensor,
    mixing: MhcMixing<'t>,
    norm: &'t Tensor,
    norm_eps: f32,
}

impl BoundaryCall<'_> {
    const OWNER: &'static str = "mhc_boundary";
}

impl Fused for BoundaryCall<'_> {
    type Output = (Tensor, Tensor, MhcMixes);

    fn device(&self) -> &Device {
        self.streams.device()
    }

    fn cpu(self) -> Result<Self::Output> {
        let Self {
            streams,
            pending,
            pre,
            mixing,
            norm,
            norm_eps,
        } = self;
        let reached = match pending {
            Some((output, previous)) => post_formula(streams, output, previous)?,
            None => streams.clone(),
        };
        let mixes = mixes_formula(&reached, mixing)?;
        let input = pre_formula(&reached, pre, norm, norm_eps)?;

        Ok((reached, input, mixes))
    }

    fn launch(self) -> Result<Self::Output> {
        let Self {
            streams,
            pending,
            pre,
            mixing,
            norm,
            norm_eps,
        } = self;
        let (tokens, count, width) = extents(Self::OWNER, streams)?;
        let dtype = streams.dtype();
        let fresh = |dims: &[usize], dtype: DType| ops::empty(dims, dtype);

        let reached = match pending {
            Some(_) => fresh(&[tokens, count, width], dtype)?,
            None => streams.clone(),
        };
        let input = fresh(&[tokens, width], dtype)?;
        let mixes = MhcMixes {
            pre: fresh(&[tokens, count], DType::F32)?,
            post: fresh(&[tokens, count], DType::F32)?,
            comb: fresh(&[tokens, count, count], DType::F32)?,
        };
        if tokens == 0 {
            return Ok((reached, input, mixes));
        }

        let descriptor = BoundaryDescriptor::new(
            dtype,
            extent(Self::OWNER, count)?,
            extent(Self::OWNER, width)?,
            mixing.iterations,
            pending.is_some(),
            tokens as u64,
        );
        // A split launch's partials and arrival counters; the counters start at zero, and the
        // kernel leaves them there.
        let splits = descriptor.splits(tokens as u64) as usize;
        let (partials, counters) = if splits > 1 {
            let counters = fresh(
                &[descriptor.token_blocks(tokens as u64) as usize],
                DType::I32,
            )?;
            counters.zero_set()?;
            let partials = fresh(
                &[splits, tokens, descriptor.partial_width() as usize],
                DType::F32,
            )?;
            (Some(partials), Some(counters))
        } else {
            (None, None)
        };

        let pending = match pending {
            Some((output, previous)) => Some(Pending {
                output: ops::operand(&output.contiguous()?)?,
                post: ops::operand(&previous.post.contiguous()?)?,
                comb: ops::operand(&previous.comb.contiguous()?)?,
                streams: ops::operand(&reached)?,
            }),
            None => None,
        };
        ops::launch::<MhcBoundary>(
            &descriptor,
            BoundaryArgs {
                streams: ops::operand(&streams.contiguous()?)?,
                pending,
                weight: ops::operand(mixing.projection)?,
                scale: ops::operand(mixing.scale)?,
                base: ops::operand(mixing.base)?,
                pre: ops::operand(&pre.contiguous()?)?,
                norm: ops::operand(norm)?,
                input: ops::operand(&input)?,
                next_pre: ops::operand(&mixes.pre)?,
                next_post: ops::operand(&mixes.post)?,
                next_comb: ops::operand(&mixes.comb)?,
                partials: partials.as_ref().map(ops::operand).transpose()?,
                counters: counters.as_ref().map(ops::operand).transpose()?,
                rms_eps: mixing.rms_eps,
                eps: mixing.eps,
                norm_eps,
            },
        )?;
        Ok((reached, input, mixes))
    }
}

/// A sub-layer's `output`, `[tokens, d]`, written back into `streams`, `[tokens, n, d]`, with
/// the post-mix and the combination matrix of the `mixes` its input was read with:
/// `h'_k = round(post_k · y + Σ_j comb[j, k] · h_j)`, `[tokens, n, d]` in the streams' type.
///
/// # Errors
///
/// When an operand has other extents or types, or the operator refuses its arguments.
pub fn mhc_post(streams: &Tensor, output: &Tensor, mixes: &MhcMixes) -> Result<Tensor> {
    const OWNER: &str = PostCall::OWNER;

    let (tokens, count, width) = extents(OWNER, streams)?;
    exactly(OWNER, "output", output, &[tokens, width], streams.dtype())?;
    mixes_of(OWNER, mixes, tokens, count)?;

    compute(PostCall {
        streams,
        output,
        mixes,
    })
}

/// A checked call of [`mhc_post`].
struct PostCall<'t> {
    streams: &'t Tensor,
    output: &'t Tensor,
    mixes: &'t MhcMixes,
}

impl PostCall<'_> {
    const OWNER: &'static str = "mhc_post";
}

impl Fused for PostCall<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.streams.device()
    }

    fn cpu(self) -> Result<Tensor> {
        post_formula(self.streams, self.output, self.mixes)
    }

    fn launch(self) -> Result<Tensor> {
        let Self {
            streams,
            output,
            mixes,
        } = self;
        let (tokens, count, width) = extents(Self::OWNER, streams)?;
        let out = ops::empty(streams.dims(), streams.dtype())?;
        if tokens == 0 {
            return Ok(out);
        }

        ops::launch::<MhcPost>(
            &PostDescriptor::new(
                streams.dtype(),
                extent(Self::OWNER, count)?,
                extent(Self::OWNER, width)?,
            ),
            PostArgs {
                streams: ops::operand(&streams.contiguous()?)?,
                output: ops::operand(&output.contiguous()?)?,
                post: ops::operand(&mixes.post.contiguous()?)?,
                comb: ops::operand(&mixes.comb.contiguous()?)?,
                out: ops::operand(&out)?,
            },
        )?;
        Ok(out)
    }
}

/// `streams`, `[tokens, n, d]`, read into one row with the pre-mix `pre`, `[tokens, n]` FP32, and
/// RMS-normalized with the gain `norm`, `[d]`: `rms_norm(round(Σ_j pre_j · h_j)) · w`,
/// `[tokens, d]` in the streams' type.
///
/// # Errors
///
/// When an operand has other extents or types, or the operator refuses its arguments.
pub fn mhc_pre(streams: &Tensor, pre: &Tensor, norm: &Tensor, eps: f32) -> Result<Tensor> {
    const OWNER: &str = PreCall::OWNER;

    let (tokens, count, width) = extents(OWNER, streams)?;
    exactly(OWNER, "pre-mix", pre, &[tokens, count], DType::F32)?;
    exactly(OWNER, "norm", norm, &[width], streams.dtype())?;

    compute(PreCall {
        streams,
        pre,
        norm,
        eps,
    })
}

/// A checked call of [`mhc_pre`].
struct PreCall<'t> {
    streams: &'t Tensor,
    pre: &'t Tensor,
    norm: &'t Tensor,
    eps: f32,
}

impl PreCall<'_> {
    const OWNER: &'static str = "mhc_pre";
}

impl Fused for PreCall<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.streams.device()
    }

    fn cpu(self) -> Result<Tensor> {
        pre_formula(self.streams, self.pre, self.norm, self.eps)
    }

    fn launch(self) -> Result<Tensor> {
        let Self {
            streams,
            pre,
            norm,
            eps,
        } = self;
        let (tokens, count, width) = extents(Self::OWNER, streams)?;
        let out = ops::empty((tokens, width), streams.dtype())?;
        if tokens == 0 {
            return Ok(out);
        }

        ops::launch::<MhcPre>(
            &PreDescriptor::new(
                streams.dtype(),
                extent(Self::OWNER, count)?,
                extent(Self::OWNER, width)?,
            ),
            PreArgs {
                streams: ops::operand(&streams.contiguous()?)?,
                pre: ops::operand(&pre.contiguous()?)?,
                norm: ops::operand(norm)?,
                out: ops::operand(&out)?,
                eps,
            },
        )?;
        Ok(out)
    }
}

/// The CPU formula of [`mhc_post`]: the sum over the read streams first, each product rounded,
/// then the post term added, all in FP32, rounded once.
fn post_formula(streams: &Tensor, output: &Tensor, mixes: &MhcMixes) -> Result<Tensor> {
    let (tokens, count, width) = extents(PostCall::OWNER, streams)?;
    let read = streams.to_dtype(ACCUMULATE)?;
    let comb = mixes.comb.to_dtype(ACCUMULATE)?;

    let mut mixed = Tensor::zeros((tokens, count, width), ACCUMULATE, streams.device())?;
    for j in 0..count {
        // comb[:, j, k] · h_j for every written stream k.
        let weights = comb.narrow(1, j, 1)?.reshape((tokens, count, 1))?;
        let stream = read.narrow(1, j, 1)?;
        mixed = (mixed + weights.broadcast_mul(&stream)?)?;
    }
    let gained = mixes
        .post
        .to_dtype(ACCUMULATE)?
        .unsqueeze(2)?
        .broadcast_mul(&output.to_dtype(ACCUMULATE)?.unsqueeze(1)?)?;

    Ok((gained + mixed)?.to_dtype(streams.dtype())?)
}

/// The CPU formula of a connection's mixes: the projection of the flattened streams scaled by
/// their reciprocal RMS, then each group's scale and bias, the sigmoids and the Sinkhorn
/// iterations.
fn mixes_formula(streams: &Tensor, mixing: MhcMixing<'_>) -> Result<MhcMixes> {
    let (tokens, count, width) = extents(BoundaryCall::OWNER, streams)?;
    let flat = streams
        .reshape((tokens, count * width))?
        .to_dtype(ACCUMULATE)?;
    let inverse = (flat.sqr()?.mean_keepdim(1)? + f64::from(mixing.rms_eps))?
        .sqrt()?
        .recip()?;
    let mixes = flat
        .matmul(&mixing.projection.t()?.contiguous()?)?
        .broadcast_mul(&inverse)?;

    let scale = mixing.scale.to_vec1::<f32>()?;
    let base = mixing.base;
    let group = |index: usize, first: usize, length: usize| -> Result<Tensor> {
        Ok(mixes
            .narrow(1, first, length)?
            .affine(f64::from(scale[index]), 0.0)?
            .broadcast_add(&base.narrow(0, first, length)?)?)
    };
    let eps = f64::from(mixing.eps);

    let pre = (group(0, 0, count)?.sigmoid()? + eps)?;
    let post = group(1, count, count)?.sigmoid()?.affine(2.0, 0.0)?;

    // Row softmax plus ε, a column normalization, then the alternating iterations.
    let logits = group(2, 2 * count, count * count)?.reshape((tokens, count, count))?;
    let shifted = logits
        .broadcast_sub(&logits.max_keepdim(D::Minus1)?)?
        .exp()?;
    let mut comb = (shifted.broadcast_div(&shifted.sum_keepdim(D::Minus1)?)? + eps)?;
    comb = comb.broadcast_div(&(comb.sum_keepdim(1)? + eps)?)?;
    for _ in 1..mixing.iterations {
        comb = comb.broadcast_div(&(comb.sum_keepdim(D::Minus1)? + eps)?)?;
        comb = comb.broadcast_div(&(comb.sum_keepdim(1)? + eps)?)?;
    }

    Ok(MhcMixes { pre, post, comb })
}

/// The CPU formula of [`mhc_pre`]: the collapse in FP32, each product rounded, rounded into the
/// streams' type, then normalized in FP32 and rounded again.
fn pre_formula(streams: &Tensor, pre: &Tensor, norm: &Tensor, eps: f32) -> Result<Tensor> {
    let (tokens, count, width) = extents(PreCall::OWNER, streams)?;
    let read = streams.to_dtype(ACCUMULATE)?;
    let pre = pre.to_dtype(ACCUMULATE)?;

    let mut collapsed = Tensor::zeros((tokens, width), ACCUMULATE, streams.device())?;
    for j in 0..count {
        let term = pre
            .narrow(1, j, 1)?
            .broadcast_mul(&read.narrow(1, j, 1)?.reshape((tokens, width))?)?;
        collapsed = (collapsed + term)?;
    }
    let collapsed = collapsed.to_dtype(streams.dtype())?.to_dtype(ACCUMULATE)?;

    let inverse = (collapsed.sqr()?.mean_keepdim(1)? + f64::from(eps))?
        .sqrt()?
        .recip()?;
    Ok(collapsed
        .broadcast_mul(&inverse)?
        .broadcast_mul(&norm.to_dtype(ACCUMULATE)?)?
        .to_dtype(streams.dtype())?)
}

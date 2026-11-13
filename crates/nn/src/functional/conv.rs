//! Convolutions with torch's arguments, and the causal padding of a video convolution's input.
//!
//! ```text
//! conv1d            x [n, c, l]        weight [k, c/groups, t]         out [n, k, l']
//! conv_transpose1d  x [n, c, l]        weight [c, k/groups, t]         out [n, k, l']
//! conv3d            x [n, c, d, h, w]  weight [k, c/groups, t, r, s]   out [n, k, d', h', w']
//! ```
//!
//! Each is one launch of its cuDNN operator with the bias and the groups inside it, where the
//! tensor library's own convolution launches one operator per group and adds no bias. The
//! output extent follows torch's rule along every spatial axis: `(l + 2p − d(t − 1) − 1) / s + 1`
//! for a convolution and `(l − 1)s − 2p + d(t − 1) + 1` for a transposed one, which has no
//! output padding. Every sum is formed in FP32 and rounds once into the input's element type;
//! `tf32` is a model's declaration that cuDNN may narrow the products of an FP32 convolution.
//!
//! ## Storage order
//!
//! A 3-d convolution reads its input in the order the tensor stores it. A tensor allocated
//! `[n, d, h, w, c]` and permuted to the logical `[n, c, d, h, w]` is stored channels-last, and
//! the convolution then writes its output stored the same way; any other strided view is copied
//! into the packed channel-first order first. The filter is always the checkpoint's packed
//! `[k, c/groups, t, r, s]`: a storage order is the activations', never the weights'.
//!
//! ## Causal padding
//!
//! A causal video convolution reads an input padded beyond what a zero padding states: the
//! spatial border mirrored or replicated ([`Padding`]) and zero frames in front, so no output
//! frame reads a later input frame. [`causal_pad`] is that padding as a concatenation of views;
//! the normalization that precedes such a convolution writes its output already padded
//! ([`frame_norm_pad`](super::frame_norm_pad)).

use ops::conv::{Args, Conv1d, Conv3d, ConvTranspose1d, Descriptor};
use ops::frame::is_channels_last;
use tensor::{DType, Device, Tensor};

use super::on_cpu;
use crate::error::{Error, Result};

/// How a causal pad fills the rows and columns beyond a frame's edge: torch's `padding_mode`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Padding {
    /// Mirror the rows and columns inside the edge (`reflect`).
    Reflect,
    /// Repeat the edge row and column (`replicate`).
    Replicate,
}

/// `values` over the operator's three spatial axes: the axes past `N` take `fill`.
fn axes<const N: usize>(values: [u32; N], fill: u32) -> [u32; 3] {
    let mut axes = [fill; 3];
    axes[..N].copy_from_slice(&values);
    axes
}

/// `x`, `[n, c, l]`, convolved with `weight`, `[k, c/groups, t]`, plus `bias`, `[k]`: torch's
/// `conv1d`, `[n, k, (l + 2·padding − dilation·(t − 1) − 1) / stride + 1]` in `x`'s element
/// type.
///
/// # Errors
///
/// When the operands are not a sequence and its filter, the filter does not divide into
/// `groups`, the bias is not one element per output channel in `x`'s element type, the input
/// leaves no output, or the operator refuses its arguments.
#[expect(
    clippy::too_many_arguments,
    reason = "torch's conv1d arguments and the TF32 declaration"
)]
pub fn conv1d(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: u32,
    padding: u32,
    dilation: u32,
    groups: u32,
    tf32: bool,
) -> Result<Tensor> {
    convolve(
        "conv1d",
        x,
        weight,
        bias,
        [axes([stride], 1), axes([padding], 0), axes([dilation], 1)],
        groups,
        tf32,
        1,
        false,
    )
}

/// `x`, `[n, c, l]`, convolved transposed with `weight`, `[c, k/groups, t]`, plus `bias`, `[k]`:
/// torch's `conv_transpose1d` without output padding,
/// `[n, k, (l − 1)·stride − 2·padding + dilation·(t − 1) + 1]` in `x`'s element type.
///
/// # Errors
///
/// As [`conv1d`].
#[expect(
    clippy::too_many_arguments,
    reason = "torch's conv_transpose1d arguments and the TF32 declaration"
)]
pub fn conv_transpose1d(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: u32,
    padding: u32,
    dilation: u32,
    groups: u32,
    tf32: bool,
) -> Result<Tensor> {
    convolve(
        "conv_transpose1d",
        x,
        weight,
        bias,
        [axes([stride], 1), axes([padding], 0), axes([dilation], 1)],
        groups,
        tf32,
        1,
        true,
    )
}

/// `x`, `[n, c, d, h, w]`, convolved with `weight`, `[k, c/groups, t, r, s]`, plus `bias`,
/// `[k]`: torch's `conv3d`, `[n, k, d', h', w']` in `x`'s element type and storage order (module
/// documentation). `stride`, `padding` and `dilation` are per axis, depth first.
///
/// # Errors
///
/// As [`conv1d`], for a frame and its filter.
#[expect(
    clippy::too_many_arguments,
    reason = "torch's conv3d arguments and the TF32 declaration"
)]
pub fn conv3d(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: [u32; 3],
    padding: [u32; 3],
    dilation: [u32; 3],
    groups: u32,
    tf32: bool,
) -> Result<Tensor> {
    convolve(
        "conv3d",
        x,
        weight,
        bias,
        [stride, padding, dilation],
        groups,
        tf32,
        3,
        false,
    )
}

/// The convolution of `rank` spatial axes the three functions share: the checks, torch's output
/// extents, the formula on CPU tensors and the one launch on tensors of the device backend.
#[expect(
    clippy::too_many_arguments,
    reason = "the three public convolutions' arguments, with what tells them apart"
)]
fn convolve(
    owner: &'static str,
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    [stride, padding, dilation]: [[u32; 3]; 3],
    groups: u32,
    tf32: bool,
    rank: usize,
    transposed: bool,
) -> Result<Tensor> {
    let (dims, filter) = (x.dims(), weight.dims());
    if dims.len() != rank + 2 || filter.len() != rank + 2 {
        return Err(Error::shape(
            owner,
            format!(
                "an input of {dims:?} and a filter of {filter:?} for a convolution over {rank} \
                 axes"
            ),
        ));
    }

    // The filter is `[out, in / groups, taps…]`, and torch stores a transposed one
    // `[in, out / groups, taps…]`.
    let group_count = groups as usize;
    let (in_channels, out_channels) = match transposed {
        false => (filter[1] * group_count, filter[0]),
        true => (filter[0], filter[1] * group_count),
    };
    let positive = |values: &[u32; 3]| values.iter().all(|value| *value > 0);
    if groups == 0
        || in_channels == 0
        || out_channels == 0
        || dims[1] != in_channels
        || !in_channels.is_multiple_of(group_count)
        || !out_channels.is_multiple_of(group_count)
        || filter[2..].contains(&0)
        || !positive(&stride)
        || !positive(&dilation)
    {
        return Err(Error::shape(
            owner,
            format!(
                "an input of {dims:?} and a filter of {filter:?} in {groups} groups, with stride \
                 {stride:?} and dilation {dilation:?}, are not a convolution"
            ),
        ));
    }
    if let Some(bias) = bias
        && (bias.dims() != [out_channels] || bias.dtype() != x.dtype())
    {
        return Err(Error::shape(
            owner,
            format!(
                "a bias of {} {:?} for {out_channels} output channels of {}",
                ops::abi::name(bias.dtype()),
                bias.dims(),
                ops::abi::name(x.dtype())
            ),
        ));
    }

    let mut kernel = [1_u32; 3];
    let mut out = vec![dims[0], out_channels];
    for axis in 0..rank {
        let taps = filter[axis + 2];
        kernel[axis] = u32::try_from(taps)
            .map_err(|_| Error::shape(owner, format!("a filter of {taps} taps")))?;

        let length = dims[axis + 2];
        let span = dilation[axis] as usize * (taps - 1);
        let (step, pad) = (stride[axis] as usize, padding[axis] as usize);
        let extent = match transposed {
            true => ((length.max(1) - 1) * step + span + 1).checked_sub(2 * pad),
            false => (length + 2 * pad)
                .checked_sub(span + 1)
                .map(|reach| reach / step + 1),
        };

        out.push(extent.filter(|extent| *extent > 0).ok_or_else(|| {
            Error::shape(
                owner,
                format!("an input of {length} along axis {axis} leaves no output"),
            )
        })?);
    }

    if on_cpu(x) {
        let (x32, weight32) = (x.to_dtype(DType::F32)?, weight.to_dtype(DType::F32)?);
        let window = |axis: usize| {
            (
                padding[axis] as usize,
                stride[axis] as usize,
                dilation[axis] as usize,
            )
        };

        let mut sum = match (rank, transposed) {
            (1, false) => {
                let (padding, stride, dilation) = window(0);
                x32.conv1d(&weight32, padding, stride, dilation, group_count)?
            }
            (1, true) => {
                let (padding, stride, dilation) = window(0);
                x32.conv_transpose1d(&weight32, padding, 0, stride, dilation, group_count)?
            }
            _ => conv3d_formula(
                &x32,
                &weight32,
                [stride, padding, dilation],
                group_count,
                &out,
            )?,
        };
        if let Some(bias) = bias {
            // One bias per output channel, over every spatial axis.
            let mut per_channel = vec![1; rank + 2];
            per_channel[1] = out_channels;
            sum = sum.broadcast_add(&bias.to_dtype(DType::F32)?.reshape(per_channel)?)?;
        }
        return Ok(sum.to_dtype(x.dtype())?);
    }

    // The operator derives every operand's strides from the storage order it is told: packed
    // channel-first, or channels-last for a frame stored that way.
    let channels_last = rank == 3 && is_channels_last(&backend::operand(x)?);
    let input = if channels_last { x.clone() } else { packed(x)? };
    let weight = packed(weight)?;
    let result = if channels_last {
        backend::empty((out[0], out[2], out[3], out[4], out[1]), x.dtype())?
            .permute((0, 4, 1, 2, 3))?
    } else {
        backend::empty(out, x.dtype())?
    };

    let descriptor = Descriptor {
        input: x.dtype(),
        weight: weight.dtype(),
        output: x.dtype(),
        kernel,
        stride,
        padding,
        dilation,
        groups,
        bias: bias.is_some(),
        tf32,
        channels_last,
        rank: rank as u32,
        transposed,
    };
    let args = Args {
        input: backend::operand(&input)?,
        weight: backend::operand(&weight)?,
        bias: bias.map(backend::operand).transpose()?,
        out: backend::operand(&result)?,
    };
    match (rank, transposed) {
        (1, false) => backend::launch::<Conv1d>(&descriptor, args)?,
        (1, true) => backend::launch::<ConvTranspose1d>(&descriptor, args)?,
        _ => backend::launch::<Conv3d>(&descriptor, args)?,
    }
    Ok(result)
}

/// `tensor` stored packed row-major: itself when it is, and a copy otherwise.
pub(super) fn packed(tensor: &Tensor) -> Result<Tensor> {
    let operand = backend::operand(tensor)?;

    if operand.strides == ops::Strides::row_major(operand.shape) {
        Ok(tensor.clone())
    } else {
        Ok(tensor.force_contiguous()?)
    }
}

/// The 3-d convolution of the FP32 CPU tensors `x` and `weight` into `out` extents, by the
/// definition: each output element sums its group's input channels over the filter's taps, a
/// tap outside the input reading zero.
fn conv3d_formula(
    x: &Tensor,
    weight: &Tensor,
    [stride, padding, dilation]: [[u32; 3]; 3],
    groups: usize,
    out: &[usize],
) -> Result<Tensor> {
    let (batches, _, depth, height, width) = x.dims5()?;
    let (out_channels, group_channels, taps_d, taps_h, taps_w) = weight.dims5()?;
    let (out_d, out_h, out_w) = (out[2], out[3], out[4]);
    let in_channels = group_channels * groups;

    let input: Vec<f32> = x.flatten_all()?.to_vec1()?;
    let filter: Vec<f32> = weight.flatten_all()?.to_vec1()?;

    // The input position a tap of an output position reads along one axis, when it is inside.
    let source = |position: usize, tap: usize, axis: usize, length: usize| {
        (position * stride[axis] as usize + tap * dilation[axis] as usize)
            .checked_sub(padding[axis] as usize)
            .filter(|index| *index < length)
    };

    let mut values = Vec::with_capacity(out.iter().product());
    for batch in 0..batches {
        for channel in 0..out_channels {
            let first = channel / (out_channels / groups) * group_channels;

            for z in 0..out_d {
                for y in 0..out_h {
                    for x in 0..out_w {
                        let mut sum = 0.0_f32;

                        for tap_c in 0..group_channels {
                            for tap_d in 0..taps_d {
                                let Some(source_d) = source(z, tap_d, 0, depth) else {
                                    continue;
                                };
                                for tap_h in 0..taps_h {
                                    let Some(source_h) = source(y, tap_h, 1, height) else {
                                        continue;
                                    };
                                    for tap_w in 0..taps_w {
                                        let Some(source_w) = source(x, tap_w, 2, width) else {
                                            continue;
                                        };

                                        let element = (((batch * in_channels + first + tap_c)
                                            * depth
                                            + source_d)
                                            * height
                                            + source_h)
                                            * width
                                            + source_w;
                                        let tap = (((channel * group_channels + tap_c) * taps_d
                                            + tap_d)
                                            * taps_h
                                            + tap_h)
                                            * taps_w
                                            + tap_w;
                                        sum += input[element] * filter[tap];
                                    }
                                }
                            }
                        }
                        values.push(sum);
                    }
                }
            }
        }
    }
    Ok(Tensor::from_vec(values, out, &Device::Cpu)?)
}

/// `x` padded along `axis` by `before` and `after` elements filled as `padding` states: the
/// elements inside the edge mirrored, or the edge element repeated.
fn pad_edge(
    x: &Tensor,
    axis: usize,
    before: usize,
    after: usize,
    padding: Padding,
) -> Result<Tensor> {
    if before == 0 && after == 0 {
        return Ok(x.clone());
    }

    let length = x.dims()[axis];
    if padding == Padding::Reflect && before.max(after) >= length {
        return Err(Error::shape(
            "causal_pad",
            format!(
                "an axis of {length} elements mirrors fewer than {} beyond its edge",
                before.max(after)
            ),
        ));
    }

    // Element `i` before the edge is the mirror of element `i` inside it, or the edge itself.
    let leading = (0..before).map(|index| match padding {
        Padding::Reflect => before - index,
        Padding::Replicate => 0,
    });
    let trailing = (0..after).map(|index| match padding {
        Padding::Reflect => length - 2 - index,
        Padding::Replicate => length - 1,
    });

    let mut pieces = Vec::with_capacity(before + after + 1);
    for index in leading {
        pieces.push(x.narrow(axis, index, 1)?);
    }
    pieces.push(x.clone());
    for index in trailing {
        pieces.push(x.narrow(axis, index, 1)?);
    }
    Ok(Tensor::cat(&pieces, axis)?)
}

/// The causal padding of a video convolution's input: `x`, `[n, c, d, h, w]`, with `top`,
/// `bottom`, `left` and `right` (`spatial`, in that order) rows and columns around each frame,
/// filled as `padding` states, and `front` zero frames before the first.
///
/// The frames in front are zeros and none follow the source, so no output frame of the
/// convolution that reads the padded input depends on a later input frame. The result is
/// `[n, c, d + front, h + top + bottom, w + left + right]` in `x`'s element type: a
/// concatenation of views of `x`, on the CPU and on a device alike.
///
/// # Errors
///
/// When `x` is not a frame, or a mirrored border is as wide as the frame.
pub fn causal_pad(
    x: &Tensor,
    spatial: [usize; 4],
    front: usize,
    padding: Padding,
) -> Result<Tensor> {
    if x.rank() != 5 {
        return Err(Error::shape(
            "causal_pad",
            format!("a frame is rank 5, not {:?}", x.dims()),
        ));
    }
    let [top, bottom, left, right] = spatial;

    let rows = pad_edge(x, 3, top, bottom, padding)?;
    let columns = pad_edge(&rows, 4, left, right, padding)?;
    Ok(columns.pad_with_zeros(2, front, 0)?)
}

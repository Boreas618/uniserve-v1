//! The fused operators and the stateless functions: one function each, over tensors.
//!
//! A fused operator is a computation the kernel library has one kernel for where a composition of
//! tensor operations would launch several and round between them: a normalization with its row
//! statistics, a gated activation, a product over packed weights, the routed products of a
//! mixture of experts. Each is one function that checks its call and states it as a value of a
//! call type implementing [`Fused`], whose two implementations state the same arithmetic on the
//! two kinds of device a tensor is on:
//!
//! - the **CPU device** computes the operator's formula with the tensor library's operations, in
//!   FP32, rounding where the kernel rounds ([`Fused::cpu`]);
//! - the **device backend** takes the outputs from the context's allocator (`ops::empty`),
//!   states the arguments (`ops::operand`) and launches the operator (`ops::launch`), with the
//!   descriptor the operator derives from the call ([`Fused::launch`]). Under a meta context the
//!   launch states its key and computes nothing.
//!
//! [`compute`] is the one place the implementation is chosen, by the device of the call's tensors;
//! mixing devices is refused by the tensor operation or the operand that meets the other device.
//!
//! The device backend's arguments are device operands without the tensors they view, and its
//! launches are capture-safe calls of compiled kernels recorded in launch tables; the CPU device
//! computes new tensors through the tensor library's own CPU backend. A call of several launches
//! (an encoding and the product that reads it, a normalization that narrows into another
//! operator's output) states its mathematics once on the CPU, as the composition the function
//! promises. So both implementations are the function's, beside each other, and not a CPU
//! operator of `ops` per kernel.
//!
//! What is not here is every computation that is a composition of generic tensor operations: a
//! transpose, an affine form, a residual add, a cast, a row gather. A model writes those as
//! tensor expressions. A plain activation the tensor library has ([`silu`], [`gelu`]) is such an
//! operation; it is stated here by name for the modules that take an [`Activation`].
//!
//! ## Rows
//!
//! A row-wise function reads its input as `[rows, columns]`: the last axis is the row, and every
//! leading axis counts rows. A tensor of more axes is viewed as its rows and the result takes its
//! leading axes back.
//!
//! ## Kernel facts
//!
//! The descriptor values that belong to a kernel package and not to the model, such as the row
//! tile a normalization loads, the elements one activation program handles, the tile and
//! workspace of a grouped product or whether a launch is a programmatic dependent launch, are
//! the operator's: each `ops` descriptor derives them from the call (`Descriptor::new` or
//! `Descriptor::of`), so a function states only its call's mathematics and every launch site of
//! an operator states the same key.

mod activation;
mod activation1d;
mod attention;
mod commit;
mod conv;
mod embedding;
mod frame;
mod indexed;
mod linear;
mod logits;
mod moe;
mod noise;
mod norm;
mod nvfp4;
mod pool;
mod recurrent;
mod residual_norm;
mod rotary;
mod rows;
mod sampling;
mod sparse;
mod vsa;

use tensor::{DType, Device, Shape, Tensor};

use crate::error::{Error, Result};

pub use activation::{
    Activation, Rounding, act_and_mul, gelu, gelu_tanh, silu, snake, swiglu, value_first_swiglu,
};
pub use activation1d::activation1d;
pub use attention::{
    dense_attention, dense_attention_varlen, kv_write_, paged_attention, zero_arrival_counters_,
};
pub use commit::slot_commit_;
pub use conv::{Padding, ProductPrecision, causal_pad, conv_transpose1d, conv1d, conv3d};
pub use embedding::{embedding, vocabulary_shard};
pub use frame::frame_norm_pad;
pub use indexed::{index_scores, indexed_attention};
pub use linear::{WeightOrder, dynamic_scales, linear, linear_fp8, linear_nvfp4};
pub use logits::{LOGITS, vocab_unpad};
pub use moe::{
    EXPANDED_TO_PERMUTED, NUM_NON_EXITING_TILES, PERMUTED_TO_EXPANDED, PERMUTED_TO_TOKEN,
    SORT_MAPS, TILE_TO_EXPERT, TILE_TO_LIMIT, TOTAL_PADDED_TOKENS, expert_combine, expert_dispatch,
    expert_exchange_fused, gather_gemm, gather_gemm_nvfp4, moe_sort, moe_unpermute, route_gemm,
    route_gemm_nvfp4, sigmoid_gated_add_, topk_softmax,
};
pub use noise::{philox_normal, philox_normal_like, philox_normal_rows};
pub use norm::{add_rms_norm, gated_rms_norm, layer_norm, rms_norm};
pub use nvfp4::nvfp4_quantize;
pub use pool::adaptive_avg_pool2d;
pub use recurrent::{StateAxes, causal_conv1d, gated_delta_rule};
pub use residual_norm::{
    modulated_rms_norm, modulated_rms_norm_, scaled_residual_layer_norm, scaled_residual_rms_norm_,
};
pub use rotary::{
    grid_positions, qk_norm_rope, qk_norm_rope_, rotary, rotary_factors, rotary_trailing_,
};
pub use rows::place_rows_;
pub use sampling::{
    argmax, log_softmax, logits_shape, logprob_gather, sampling_from_logits, softmax, top_k,
    top_logprobs,
};
pub use sparse::sparse_attention;
pub use vsa::{
    VSA_BLOCK_ROWS, vsa_add_gated_tiles_, vsa_compose_gated_rows, vsa_compress_pooled,
    vsa_prepare_tiles_, vsa_scores, vsa_select_tiles, vsa_zero_padding_,
};

/// The accumulation element type of every floating-point operator here.
pub(crate) const ACCUMULATE: DType = DType::F32;

/// One checked call of a fused operator, with its implementation on each kind of device.
///
/// A function checks its arguments once, then states the call as a value of a type implementing
/// this trait and hands it to [`compute`]. Both implementations keep the function's contract: the
/// same output shapes and element types, and the same rounding points.
pub(crate) trait Fused {
    /// What the call returns.
    type Output;

    /// The device the call's tensors are on.
    fn device(&self) -> &Device;

    /// The call on CPU tensors: the operator's formula in the tensor library's operations.
    fn cpu(self) -> Result<Self::Output>;

    /// The call on tensors of the device backend: the operator's launches over outputs from the
    /// context's allocator.
    fn launch(self) -> Result<Self::Output>;
}

/// Computes `call` by the implementation of its tensors' device: the one place a fused operator's
/// implementation is chosen.
pub(crate) fn compute<F: Fused>(call: F) -> Result<F::Output> {
    match call.device() {
        Device::Cpu => call.cpu(),
        Device::Backend(_) => call.launch(),
    }
}

/// The width of `x`'s rows, its last extent, as a descriptor states it.
pub(crate) fn columns(owner: &'static str, x: &Tensor) -> Result<u32> {
    let Some(last) = x.dims().last() else {
        return Err(Error::shape(
            owner,
            "a scalar has no rows to compute on".to_owned(),
        ));
    };

    u32::try_from(*last)
        .ok()
        .filter(|columns| *columns > 0)
        .ok_or_else(|| Error::shape(owner, format!("rows of {last} elements")))
}

/// `x` as its rows, `[rows, columns]`: itself when it has two axes, and a view of the same
/// elements otherwise (a copy when `x` is a strided view that has no such view).
pub(crate) fn as_rows(owner: &'static str, x: &Tensor) -> Result<Tensor> {
    let width = columns(owner, x)? as usize;

    if x.rank() == 2 {
        Ok(x.clone())
    } else {
        Ok(x.reshape((x.elem_count() / width, width))?)
    }
}

/// `rows`, `[rows, width]`, under the leading axes of `like`: the shape a row-wise function
/// returns for an input of `like`'s shape.
pub(crate) fn like_rows(rows: Tensor, like: &Tensor) -> Result<Tensor> {
    if like.rank() == 2 {
        return Ok(rows);
    }

    let width = rows.dims().last().copied().unwrap_or(0);
    let mut dims = like.dims().to_vec();
    if let Some(last) = dims.last_mut() {
        *last = width;
    }
    Ok(rows.reshape(Shape::from(dims))?)
}

/// Checks that `vector`, the per-channel parameter `name`, is one vector of `width` elements.
pub(crate) fn vector(owner: &'static str, name: &str, vector: &Tensor, width: u32) -> Result<()> {
    if vector.dims() != [width as usize] {
        return Err(Error::shape(
            owner,
            format!(
                "the {name} is {:?}, and rows of {width} elements take one vector of {width}",
                vector.dims()
            ),
        ));
    }
    Ok(())
}

/// Checks that `other`, the tensor `name`, has the shape and element type of `x`.
pub(crate) fn same_rows(owner: &'static str, name: &str, other: &Tensor, x: &Tensor) -> Result<()> {
    if other.dims() != x.dims() || other.dtype() != x.dtype() {
        return Err(Error::shape(
            owner,
            format!(
                "the {name} is {} {:?} against rows of {} {:?}",
                ops::abi::name(other.dtype()),
                other.dims(),
                ops::abi::name(x.dtype()),
                x.dims()
            ),
        ));
    }
    Ok(())
}

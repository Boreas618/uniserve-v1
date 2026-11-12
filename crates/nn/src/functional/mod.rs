//! The fused operators and the stateless functions: one function each, over tensors.
//!
//! A fused operator is a computation the kernel library has one kernel for where a composition of
//! tensor operations would launch several and round between them: a normalization with its row
//! statistics, a gated activation, a product over packed weights. Each is one function with two
//! bodies that state the same arithmetic:
//!
//! - on **CPU tensors** it computes its formula with tensor operations, in FP32, rounding where
//!   the kernel rounds. This is what a host check runs and the reference a device check compares
//!   with;
//! - on **tensors of the device backend** it takes its output from the context's allocator
//!   (`backend::empty`), states its arguments (`backend::operand`) and makes one launch
//!   (`backend::launch`) of the operator, with the descriptor computed from the tensors' element
//!   types and widths. Under a meta context the launch states its key and computes nothing.
//!
//! Which body runs follows from the device of the first tensor argument; mixing devices is
//! refused by the tensor operation or the operand that meets the other device.
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
//! ## Kernel constants
//!
//! A few descriptor values belong to the kernel package and not to the model: the row tile a
//! normalization loads, the elements one activation kernel handles, whether a launch is a
//! programmatic dependent launch. Each is one rule stated once here or beside its function, so
//! every launch site of an operator states the same key.

mod activation;
mod conv;
mod embedding;
mod frame;
mod linear;
mod logits;
mod norm;
mod residual_norm;

use uniserve_tensor::{DType, Device, Shape, Tensor};

use crate::error::{Error, Result};

pub use activation::{
    Activation, Rounding, act_and_mul, gelu, gelu_tanh, silu, snake, swiglu, value_first_swiglu,
};
pub use conv::{Padding, causal_pad, conv_transpose1d, conv1d, conv3d};
pub use embedding::{embedding, vocabulary_shard};
pub use frame::frame_norm_pad;
pub(crate) use linear::PRODUCT_SCALE_LAYOUT;
pub use linear::{ENCODE_CHUNK, WeightOrder, encode_block, linear, linear_fp8, linear_nvfp4};
pub use logits::{LOGITS, vocab_unpad};
pub use norm::{add_rms_norm, gated_rms_norm, layer_norm, rms_norm};
pub use residual_norm::{
    modulated_rms_norm, modulated_rms_norm_, scaled_residual_layer_norm, scaled_residual_rms_norm_,
};

/// Whether the descriptors this crate builds are the programmatic-dependent-launch variants.
pub(crate) const PDL: bool = false;

/// The accumulation element type of every floating-point operator here.
pub(crate) const ACCUMULATE: DType = DType::F32;

/// The row width a row-wise kernel loads at a time: the width rounded up to a power of two.
pub(crate) fn row_tile(columns: u32) -> u32 {
    columns.max(1).next_power_of_two()
}

/// Whether `x` is computed on the CPU device, by a function's formula in tensor operations.
pub(crate) fn on_cpu(x: &Tensor) -> bool {
    matches!(x.device(), Device::Cpu)
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

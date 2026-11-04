//! Per-channel bias addition, optionally onto a residual, channel-first.
//!
//! A convolution's bias is added in a separate pass in the reference, not folded into the
//! convolution, and H3's video encoder and decoder rely on that: a biased convolution must equal
//! `conv(x, weight) + bias` bit for bit, including the residual shortcut both passes receive.
//!
//! ```text
//! out = (residual + residual_bias) + (values + bias)
//! ```
//!
//! The operands are NCDHW — or `[batch, channels, spatial]` with the spatial axes merged, which is
//! the same storage read as three axes — and the result is contiguous and channel-first whichever
//! storage the inputs use (`ops::bias::BiasAdd`). Both biases are `[channels]`. A residual without
//! its own bias is the common case; a residual bias without a residual is refused, because there
//! would be nothing for it to bias.

use ops::bias::{self, BiasAdd as BiasAddOp};
use ops::{Shape, Strides, TensorView};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::launch::PDL;

/// Elements one bias program adds.
const BLOCK: u32 = 1024;

/// A per-channel bias, and the residual it is added to.
#[derive(Clone, Copy, Default)]
pub struct BiasAdd;

impl BiasAdd {
    /// Add `bias` per channel to `values`, and `residual_bias` per channel to `residual`.
    ///
    /// `values`, `residual` and `out` are `[batch, channels, *spatial]`; the biases are
    /// `[channels]`. An absent residual adds nothing, an absent residual bias adds `bias`'s value
    /// nowhere — both stay out of the program's descriptor, so a call without a residual compiles
    /// the plain bias program.
    pub fn apply(
        &self,
        f: &Forward,
        values: TensorView,
        bias: TensorView,
        residual: Option<TensorView>,
        residual_bias: Option<TensorView>,
        out: TensorView,
    ) -> Result<()> {
        if residual_bias.is_some() && residual.is_none() {
            return Err(Error::shape(
                "BiasAdd",
                "a residual bias needs a residual to bias".to_owned(),
            ));
        }
        let shape = values.shape;
        if shape.rank() < 3 {
            return Err(Error::shape(
                "BiasAdd",
                format!(
                    "{:?} is not [batch, channels, *spatial]",
                    shape.dims()
                ),
            ));
        }
        let channels = u32::try_from(shape.dim(1)).map_err(|_| {
            Error::shape(
                "BiasAdd",
                format!("{} channels are wider than a descriptor holds", shape.dim(1)),
            )
        })?;
        let descriptor = bias::Descriptor {
            dtype: values.dtype,
            channels,
            block: BLOCK,
            residual: residual.is_some(),
            residual_bias: residual_bias.is_some(),
            pdl: PDL,
        };
        // An absent operand carries another view's address, so one argument list serves both
        // variants and the program never reads the unused one.
        f.operate::<BiasAddOp>(
            &descriptor,
            bias::Args {
                values: merged("values", values)?,
                bias,
                residual: merged("residual", residual.unwrap_or(values))?,
                residual_bias: residual_bias.unwrap_or(bias),
                out: merged("out", out)?,
            },
        )
    }
}

/// Read an operand as `[batch, channels, spatial]`, the three axes the program walks.
///
/// A convolution's result arrives with its spatial axes separate (NCDHW) or channels last (NDHWC);
/// either way the three axes the program names are that storage read differently, so the merge is
/// exact when the spatial axes are contiguous, which is what both layouts mean:
///
/// ```text
/// channel first (spatial stride 1):   [batch · channels · spatial, spatial, 1]
/// channels last  (channel stride 1):  [spatial · channels, 1, channels]
/// ```
fn merged(name: &'static str, view: TensorView) -> Result<TensorView> {
    let shape = view.shape;
    if shape.rank() < 3 {
        return Err(Error::shape(
            "BiasAdd",
            format!("{name} is {:?}, [batch, channels, *spatial] expected", shape.dims()),
        ));
    }
    if shape.rank() == 3 {
        return Ok(view);
    }
    let (batch, channels) = (shape.dim(0), shape.dim(1));
    let spatial: u64 = shape.dims()[2..].iter().product();
    let strides = if view.strides == Strides::row_major(shape) {
        Strides::new(&[channels * spatial, spatial, 1])
    } else if view.strides.get(1) == 1 && view.strides.get(0) == channels * spatial {
        Strides::new(&[channels * spatial, 1, channels])
    } else {
        return Err(Error::shape(
            "BiasAdd",
            format!(
                "{name} is stored neither channel-first nor channels-last across its spatial axes; \
                 found strides {}",
                view.strides
            ),
        ));
    };
    Ok(TensorView::with_strides(
        view.ptr,
        view.dtype,
        Shape::new(&[batch, channels, spatial]),
        strides,
    ))
}

//! A convolution's bias, and the residual it is added back into, in one pass.
//!
//! The reference adds a convolution's bias in a separate step rather than inside the convolution
//! (`uniserve/nn/functional/_bias.py`): `out = (residual + residual_bias) + (values + bias)`, each
//! sum rounding once in the operands' type, so a biased convolution equals `conv(x, weight) + bias`
//! bit for bit. The H3 video decoder's blocks rely on that equality (specs/worker.md 9, "bias
//! add"; `ops::bias`).
//!
//! The operands are `[batch, channels, spatial]` in any storage whose three axes the strides
//! describe — contiguous channel-first, or channels-last read as the same shape with the last two
//! strides exchanged. The result is written channel-first and contiguous, which is the layout the
//! next operator of a decoder block expects.

use ops::TensorView;
use ops::bias::{self, BiasAdd as BiasAddOp};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::launch::PDL;

/// The elements one bias program adds.
const BLOCK: u32 = 1024;

/// A per-channel bias, with an optional residual and an optional residual bias.
pub struct BiasAdd {
    channels: u32,
    residual: bool,
    residual_bias: bool,
}

impl BiasAdd {
    /// The bias of `channels` channels.
    pub fn new(channels: u64) -> Result<Self> {
        let channels = u32::try_from(channels).map_err(|_| {
            Error::shape(
                "BiasAdd",
                format!("{channels} channels are wider than a descriptor holds"),
            )
        })?;
        if channels == 0 {
            return Err(Error::shape("BiasAdd", "a bias has channels".to_owned()));
        }
        Ok(Self {
            channels,
            residual: false,
            residual_bias: false,
        })
    }

    /// The same layer folded into a residual stream, with one bias for the update.
    pub const fn with_residual(mut self) -> Self {
        self.residual = true;
        self
    }

    /// A folded update whose residual carries a bias of its own.
    pub const fn with_residual_bias(mut self) -> Self {
        self.residual = true;
        self.residual_bias = true;
        self
    }

    /// The channels this layer biases.
    pub const fn channels(&self) -> u32 {
        self.channels
    }

    /// `out = (residual + residual_bias) + (values + bias)`.
    ///
    /// `bias` is `[channels]`; `residual` and `residual_bias` are present exactly when the layer was
    /// built with them.
    pub fn forward(
        &self,
        f: &Forward,
        values: TensorView,
        bias: TensorView,
        residual: Option<TensorView>,
        residual_bias: Option<TensorView>,
        out: TensorView,
    ) -> Result<()> {
        let channels = u64::from(self.channels);
        if values.shape.rank() != 3 || values.shape.dim(1) != channels {
            return Err(Error::shape(
                "BiasAdd",
                format!(
                    "{:?} is not [batch, {channels}, spatial]",
                    values.shape.dims()
                ),
            ));
        }
        if out.shape != values.shape {
            return Err(Error::shape(
                "BiasAdd",
                format!("{:?} into {:?}", values.shape.dims(), out.shape.dims()),
            ));
        }
        if bias.shape.numel() != channels {
            return Err(Error::shape(
                "BiasAdd",
                format!("{} biases for {channels} channels", bias.shape.numel()),
            ));
        }
        let residual = match (self.residual, residual) {
            (true, Some(residual)) => residual,
            (true, None) => {
                return Err(Error::shape(
                    "BiasAdd",
                    "this layer folds into a residual, and none was given".to_owned(),
                ));
            }
            (false, Some(_)) => {
                return Err(Error::shape(
                    "BiasAdd",
                    "this layer has no residual to fold into".to_owned(),
                ));
            }
            // An absent operand carries the values' address, so one argument list serves both
            // variants; the descriptor is what decides whether it is read.
            (false, None) => values,
        };
        let residual_bias = match (self.residual_bias, residual_bias) {
            (true, Some(residual_bias)) => residual_bias,
            (true, None) => {
                return Err(Error::shape(
                    "BiasAdd",
                    "this layer biases its residual, and no bias was given".to_owned(),
                ));
            }
            (false, Some(_)) => {
                return Err(Error::shape(
                    "BiasAdd",
                    "this layer does not bias its residual".to_owned(),
                ));
            }
            (false, None) => bias,
        };
        let descriptor = bias::Descriptor {
            dtype: values.dtype,
            channels: self.channels,
            block: BLOCK,
            residual: self.residual,
            residual_bias: self.residual_bias,
            pdl: PDL,
        };
        f.operate::<BiasAddOp>(
            &descriptor,
            bias::Args {
                values,
                bias,
                residual,
                residual_bias,
                out,
            },
        )
    }
}

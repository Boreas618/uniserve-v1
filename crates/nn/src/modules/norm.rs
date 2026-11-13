//! Normalization modules: RMS normalization, its gated form, and layer normalization, each a
//! per-channel weight over the functions of [`crate::functional`].

use tensor::{DType, Module, Tensor};

use crate::error::{Error, Result};
use crate::functional;
use crate::var_builder::VarBuilder;

/// What a checkpoint states of an [`RmsNorm`], and the element type its rows are read in.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RmsNormConfig {
    /// The width of the rows it normalizes: the length of its weight.
    pub width: u32,
    /// The variance floor.
    pub eps: f32,
    /// Whether the checkpoint stores the scale zero-centred, so the row is scaled by one plus
    /// the weight (Gemma, Qwen3-Next, Qwen3.5).
    pub centred: bool,
    /// The element type the normalized rows are written in when the rows that read them are
    /// computed in a narrower type than the stream it normalizes; `None` writes the input's.
    pub output: Option<DType>,
}

/// RMS normalization with a per-channel scale: `x · rsqrt(mean(x²) + eps) · gain`
/// ([`functional::rms_norm`]).
pub struct RmsNorm {
    weight: Tensor,
    eps: f32,
    centred: bool,
    output: Option<DType>,
}

impl RmsNorm {
    /// The normalization module at `vb`: its `weight` of `config.width` channels.
    ///
    /// # Errors
    ///
    /// When the checkpoint lacks the weight or holds another width.
    pub fn load(config: &RmsNormConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        Ok(Self {
            weight: vb.get(config.width as usize, "weight")?,
            eps: config.eps,
            centred: config.centred,
            output: config.output,
        })
    }

    /// The per-channel scale.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// `x`, `[..., width]`, normalized: in the normalization's output type, or `x`'s.
    ///
    /// # Errors
    ///
    /// Rows of another width than the weight, a zero-centred gain with an output narrower than
    /// `x` (no kernel computes both), or the operator's refusal.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let output = self.output.unwrap_or(x.dtype());
        functional::rms_norm(x, &self.weight, self.eps, self.centred, output)
    }

    /// `x` added into the residual stream and the sum normalized, as one operator
    /// ([`functional::add_rms_norm`]): returns `(normalized, sum)`, where `sum` is what the
    /// stream holds from here on. A decoder's first layer normalizes its embedding alone
    /// ([`Self::forward`]) and every later layer adds into the stream first, with the same
    /// module, as the reference's `forward(x, residual)` does.
    ///
    /// # Errors
    ///
    /// A normalization loaded with a narrowing output, whose kernel takes no residual; a stream
    /// of another shape or element type than `x`; or the operator's refusal.
    pub fn forward_residual(&self, x: &Tensor, residual: &Tensor) -> Result<(Tensor, Tensor)> {
        if let Some(output) = self.output
            && output != x.dtype()
        {
            return Err(Error::shape(
                "RmsNorm",
                format!(
                    "the residual form writes its input's {} and the normalization was loaded \
                     for {}",
                    ops::abi::name(x.dtype()),
                    ops::abi::name(output)
                ),
            ));
        }
        functional::add_rms_norm(x, residual, &self.weight, self.eps, self.centred)
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> tensor::Result<Tensor> {
        Ok(Self::forward(self, x)?)
    }
}

/// What a checkpoint states of a [`GatedRmsNorm`].
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GatedRmsNormConfig {
    /// The width of the rows it normalizes.
    pub width: u32,
    /// The variance floor.
    pub eps: f32,
    /// Whether the gate passes through a sigmoid in place of the SiLU.
    pub sigmoid: bool,
}

/// RMS normalization gated by an activation of a second input:
/// `x · rsqrt(mean(x²) + eps) · weight · act(gate)`, the output norm of a gated delta network
/// ([`functional::gated_rms_norm`]).
pub struct GatedRmsNorm {
    weight: Tensor,
    eps: f32,
    sigmoid: bool,
}

impl GatedRmsNorm {
    /// The module at `vb`: its `weight` of `config.width` channels.
    ///
    /// # Errors
    ///
    /// When the checkpoint lacks the weight or holds another width.
    pub fn load(config: &GatedRmsNormConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        Ok(Self {
            weight: vb.get(config.width as usize, "weight")?,
            eps: config.eps,
            sigmoid: config.sigmoid,
        })
    }

    /// `x`, `[..., width]`, normalized and gated by `gate` of the same shape, in `x`'s element
    /// type.
    ///
    /// # Errors
    ///
    /// What the operator refuses.
    pub fn forward(&self, x: &Tensor, gate: &Tensor) -> Result<Tensor> {
        functional::gated_rms_norm(x, gate, &self.weight, self.eps, self.sigmoid)
    }
}

/// What a checkpoint states of a [`LayerNorm`].
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct LayerNormConfig {
    /// The width of the rows it normalizes.
    pub width: u32,
    /// The variance floor.
    pub eps: f32,
    /// Whether the module states a `bias` beside its `weight`.
    pub bias: bool,
}

/// Layer normalization with a per-channel scale and an optional bias:
/// `(x − mean(x)) · rsqrt(var(x) + eps) · weight + bias` ([`functional::layer_norm`]).
pub struct LayerNorm {
    weight: Tensor,
    bias: Option<Tensor>,
    eps: f32,
}

impl LayerNorm {
    /// The normalization module at `vb`: its `weight` and, when it states one, its `bias`, of
    /// `config.width` channels.
    ///
    /// # Errors
    ///
    /// When the checkpoint lacks a tensor or holds another width.
    pub fn load(config: &LayerNormConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        let width = config.width as usize;
        let bias = if config.bias {
            Some(vb.get(width, "bias")?)
        } else {
            None
        };

        Ok(Self {
            weight: vb.get(width, "weight")?,
            bias,
            eps: config.eps,
        })
    }

    /// `x`, `[..., width]`, normalized, in `x`'s element type.
    ///
    /// # Errors
    ///
    /// Rows of another width than the weight, or the operator's refusal.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        functional::layer_norm(x, &self.weight, self.bias.as_ref(), self.eps)
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> tensor::Result<Tensor> {
        Ok(Self::forward(self, x)?)
    }
}

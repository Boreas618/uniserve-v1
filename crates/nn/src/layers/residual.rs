//! The residual stream's normalization layer: a channel-scaled update folded into the residual,
//! and the normalization of the rows that follow.
//!
//! Four compositions, each one operator (specs/worker.md 9, the MiniMax-H3 table; `ops::residual`):
//! `scaled_residual`, `scaled_residual_rms_norm`, `scaled_residual_layer_norm`, `weighted_rms_norm`.
//! A block that uses them reads its residual once instead of running an add, a reduction and a
//! scale as separate launches.
//!
//! The layers own the learned parameters and the epsilon; the residual, the block's update and the
//! output are the caller's views, because the residual stream is shared by every layer of a block.

use ops::residual::{
    self, ScaledResidualLayerNorm as LayerNormOp, ScaledResidualRmsNorm as RmsNormOp,
};
use ops::{Op, TensorView};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::launch::row_tile;
use crate::weights::Weight;

/// The columns of a `[columns]` parameter, checked once.
fn columns(name: &'static str, weight: &Weight) -> Result<u32> {
    if weight.shape().rank() != 1 {
        return Err(Error::shape(
            name,
            format!(
                "a per-channel parameter is one vector, not {:?}",
                weight.shape().dims()
            ),
        ));
    }
    u32::try_from(weight.shape().numel())
        .map_err(|_| Error::shape(name, "more channels than a descriptor holds".to_owned()))
}

/// The elements one statement of a `scaled_residual` launch walks.
const BLOCK: u32 = residual::RESIDUAL_BLOCK;

/// `out = rms(x) · weight`.
pub struct WeightedRmsNorm {
    weight: Weight,
    eps: f32,
}

impl WeightedRmsNorm {
    /// The weighted normalization of `weight`'s channels.
    pub fn new(weight: Weight, eps: f32) -> Result<Self> {
        columns("WeightedRmsNorm", &weight)?;
        Ok(Self { weight, eps })
    }

    /// Channels per row.
    pub fn columns(&self) -> u32 {
        self.weight.shape().numel() as u32
    }

    /// The learned scale.
    pub fn weight(&self) -> TensorView {
        self.weight.view()
    }

    /// Normalize `x` into `out`; both are `[rows, columns]` and may alias.
    pub fn forward(&self, f: &Forward, x: TensorView, out: TensorView) -> Result<()> {
        self.run(f, x, out, None)
    }

    /// The same normalization, recording each row's largest normalized magnitude into `partials`
    /// (`[rows]` FP32) for the caller's reduction.
    pub fn forward_absmax(
        &self,
        f: &Forward,
        x: TensorView,
        out: TensorView,
        partials: TensorView,
    ) -> Result<()> {
        self.run(f, x, out, Some(partials))
    }

    fn run(
        &self,
        f: &Forward,
        x: TensorView,
        out: TensorView,
        partials: Option<TensorView>,
    ) -> Result<()> {
        let columns = self.columns();
        let descriptor = residual::WeightedRmsNormDescriptor {
            dtype: x.dtype,
            columns,
            tile: row_tile(columns),
            absmax: partials.is_some(),
        };
        let args = residual::WeightedRmsNormArgs {
            x,
            weight: self.weight.view(),
            out,
            eps: self.eps,
            partials,
        };
        let kernel = f.kernels().op::<residual::WeightedRmsNorm>(&descriptor)?;
        residual::WeightedRmsNorm::launch(kernel, &descriptor, args, f.stream).map_err(Error::from)
    }
}

/// `hidden += (update + bias) · scale`.
pub struct ScaledResidual {
    scale: Weight,
    bias: Option<Weight>,
}

impl ScaledResidual {
    /// The scaled addition of `scale`'s channels, with the update's bias when the block has one.
    pub fn new(scale: Weight, bias: Option<Weight>) -> Result<Self> {
        let channels = columns("ScaledResidual", &scale)?;
        if let Some(bias) = &bias
            && columns("ScaledResidual", bias)? != channels
        {
            return Err(Error::shape(
                "ScaledResidual",
                format!("a scale of {channels} channels and a bias of another count"),
            ));
        }
        Ok(Self { scale, bias })
    }

    /// Fold `update` into `hidden`, both `[rows, columns]`; `hidden` is written in place.
    pub fn forward(&self, f: &Forward, hidden: TensorView, update: TensorView) -> Result<()> {
        let columns = self.scale.shape().numel() as u32;
        let descriptor = residual::ScaledResidualDescriptor {
            dtype: hidden.dtype,
            columns,
            block: BLOCK,
            update_bias: self.bias.is_some(),
        };
        let args = residual::ScaledResidualArgs {
            hidden,
            update,
            update_bias: self.bias.as_ref().map(Weight::view),
            scale: self.scale.view(),
        };
        let kernel = f.kernels().op::<residual::ScaledResidual>(&descriptor)?;
        residual::ScaledResidual::launch(kernel, &descriptor, args, f.stream).map_err(Error::from)
    }
}

/// `hidden += (update + bias) · scale; out = rms(hidden) · weight`.
pub struct ScaledResidualRmsNorm {
    scale: Weight,
    bias: Option<Weight>,
    weight: Weight,
    eps: f32,
    absmax: bool,
}

impl ScaledResidualRmsNorm {
    /// The scaled addition and the RMS normalization of `weight`'s channels.
    pub fn new(scale: Weight, bias: Option<Weight>, weight: Weight, eps: f32) -> Result<Self> {
        let channels = columns("ScaledResidualRmsNorm", &weight)?;
        if columns("ScaledResidualRmsNorm", &scale)? != channels {
            return Err(Error::shape(
                "ScaledResidualRmsNorm",
                "the residual's scale and the normalization's weight differ in channels".to_owned(),
            ));
        }
        if let Some(bias) = &bias
            && columns("ScaledResidualRmsNorm", bias)? != channels
        {
            return Err(Error::shape(
                "ScaledResidualRmsNorm",
                "the update's bias and the normalization's weight differ in channels".to_owned(),
            ));
        }
        Ok(Self {
            scale,
            bias,
            weight,
            eps,
            absmax: false,
        })
    }

    /// Record each row's largest normalized magnitude on every call.
    pub fn with_absmax(mut self) -> Self {
        self.absmax = true;
        self
    }

    /// Channels per row.
    pub fn columns(&self) -> u32 {
        self.weight.shape().numel() as u32
    }

    /// The normalization's learned scale.
    pub fn weight(&self) -> TensorView {
        self.weight.view()
    }

    /// Fold `update` into `hidden` and normalize it into `out`.
    ///
    /// `hidden`, `update` and `out` are `[rows, columns]` in the activation element type; `partials`
    /// is `[rows]` FP32 and is required exactly when the layer records the magnitude.
    pub fn forward(
        &self,
        f: &Forward,
        hidden: TensorView,
        update: TensorView,
        out: TensorView,
        partials: Option<TensorView>,
    ) -> Result<()> {
        if self.absmax != partials.is_some() {
            return Err(Error::shape(
                "ScaledResidualRmsNorm",
                "the magnitude buffer is present exactly when the layer records it".to_owned(),
            ));
        }
        let columns = self.columns();
        let descriptor = residual::ScaledResidualRmsNormDescriptor {
            dtype: hidden.dtype,
            columns,
            tile: row_tile(columns),
            update_bias: self.bias.is_some(),
            absmax: self.absmax,
        };
        let args = residual::ScaledResidualRmsNormArgs {
            hidden,
            update,
            update_bias: self.bias.as_ref().map(Weight::view),
            scale: self.scale.view(),
            weight: self.weight.view(),
            out,
            eps: self.eps,
            partials,
        };
        let kernel = f.kernels().op::<RmsNormOp>(&descriptor)?;
        RmsNormOp::launch(kernel, &descriptor, args, f.stream).map_err(Error::from)
    }
}

/// `out = layer_norm(hidden + (update + bias) · scale) · weight + bias`.
pub struct ScaledResidualLayerNorm {
    scale: Weight,
    update_bias: Option<Weight>,
    weight: Weight,
    bias: Weight,
    eps: f32,
    absmax: bool,
}

impl ScaledResidualLayerNorm {
    /// The scaled addition and the layer normalization of `weight`'s channels.
    pub fn new(
        scale: Weight,
        update_bias: Option<Weight>,
        weight: Weight,
        bias: Weight,
        eps: f32,
    ) -> Result<Self> {
        let channels = columns("ScaledResidualLayerNorm", &weight)?;
        for (name, parameter) in [
            ("scale", Some(&scale)),
            ("update bias", update_bias.as_ref()),
            ("bias", Some(&bias)),
        ] {
            if let Some(parameter) = parameter
                && columns("ScaledResidualLayerNorm", parameter)? != channels
            {
                return Err(Error::shape(
                    "ScaledResidualLayerNorm",
                    format!("the {name} and the normalization's weight differ in channels"),
                ));
            }
        }
        Ok(Self {
            scale,
            update_bias,
            weight,
            bias,
            eps,
            absmax: false,
        })
    }

    /// Record each row's largest normalized magnitude on every call.
    pub fn with_absmax(mut self) -> Self {
        self.absmax = true;
        self
    }

    /// Channels per row.
    pub fn columns(&self) -> u32 {
        self.weight.shape().numel() as u32
    }

    /// Fold `update` into `hidden` and normalize it into `out`.
    pub fn forward(
        &self,
        f: &Forward,
        hidden: TensorView,
        update: TensorView,
        out: TensorView,
        partials: Option<TensorView>,
    ) -> Result<()> {
        if self.absmax != partials.is_some() {
            return Err(Error::shape(
                "ScaledResidualLayerNorm",
                "the magnitude buffer is present exactly when the layer records it".to_owned(),
            ));
        }
        let columns = self.columns();
        let descriptor = residual::ScaledResidualLayerNormDescriptor {
            dtype: hidden.dtype,
            columns,
            tile: row_tile(columns),
            update_bias: self.update_bias.is_some(),
            absmax: self.absmax,
        };
        let args = residual::ScaledResidualLayerNormArgs {
            hidden,
            update,
            update_bias: self.update_bias.as_ref().map(Weight::view),
            scale: self.scale.view(),
            weight: self.weight.view(),
            bias: self.bias.view(),
            out,
            eps: self.eps,
            partials,
        };
        let kernel = f.kernels().op::<LayerNormOp>(&descriptor)?;
        LayerNormOp::launch(kernel, &descriptor, args, f.stream).map_err(Error::from)
    }
}

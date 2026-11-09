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
use ops::{DType, TensorView};

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


    /// The descriptor one launch without a magnitude takes, for a walk without a device: rows of
    /// `dtype`, normalized into `activation`.
    pub fn descriptor_of(
        columns: u32,
        dtype: DType,
        activation: DType,
    ) -> residual::WeightedRmsNormDescriptor {
        residual::WeightedRmsNormDescriptor {
            dtype,
            columns,
            tile: row_tile(columns),
            absmax: false,
            activation,
        }
    }

    /// Normalize `x` into `out`; both are `[rows, columns]` and may alias.
    pub fn forward(&self, f: &Forward, x: TensorView, out: TensorView) -> Result<()> {
        self.run(f, x, out, None)
    }


    fn run(
        &self,
        f: &Forward,
        x: TensorView,
        out: TensorView,
        partials: Option<TensorView>,
    ) -> Result<()> {
        let mut descriptor = Self::descriptor_of(self.columns(), x.dtype, out.dtype);
        descriptor.absmax = partials.is_some();
        let args = residual::WeightedRmsNormArgs {
            x,
            weight: self.weight.view(),
            out,
            eps: self.eps,
            partials,
        };

        f.operate::<residual::WeightedRmsNorm>(&descriptor, args)
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

    /// The descriptor one launch takes, for a walk that asks a model which programs it launches with
    /// no device.
    ///
    /// `dtype` is the caller's: the hidden states' element type is the deployment's, while the
    /// layer's width and its bias's presence are its own.
    pub fn descriptor(&self, dtype: DType) -> Result<residual::ScaledResidualDescriptor> {
        Self::descriptor_of(dtype, self.scale.shape().numel(), self.bias.is_some())
    }

    /// The descriptor a residual over `columns`-wide rows launches, stated without its weights.
    ///
    /// A walk has the config's width and the checkpoint's bias presence and neither weight.
    pub fn descriptor_of(
        dtype: DType,
        columns: u64,
        update_bias: bool,
    ) -> Result<residual::ScaledResidualDescriptor> {
        Ok(residual::ScaledResidualDescriptor {
            dtype,
            columns: u32::try_from(columns).map_err(|_| {
                Error::shape(
                    "ScaledResidual",
                    "a row wider than a descriptor holds".to_owned(),
                )
            })?,
            block: BLOCK,
            update_bias,
        })
    }

    /// Fold `update` into `hidden`, both `[rows, columns]`; `hidden` is written in place.
    pub fn forward(&self, f: &Forward, hidden: TensorView, update: TensorView) -> Result<()> {
        let descriptor = self.descriptor(hidden.dtype)?;
        let args = residual::ScaledResidualArgs {
            hidden,
            update,
            update_bias: self.bias.as_ref().map(Weight::view),
            scale: self.scale.view(),
        };

        f.operate::<residual::ScaledResidual>(&descriptor, args)
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


    /// Channels per row.
    pub fn columns(&self) -> u32 {
        self.weight.shape().numel() as u32
    }


    /// Fold `update` into `hidden` and normalize it into `out`.
    ///
    /// `hidden`, `update` and `out` are `[rows, columns]` in the activation element type; `partials`
    /// is `[rows]` FP32 and is required exactly when the layer records the magnitude.
    /// The descriptor one launch takes, for a walk without a device; see
    /// [`ScaledResidual::descriptor`].
    pub fn descriptor(
        &self,
        dtype: DType,
        activation: DType,
    ) -> residual::ScaledResidualRmsNormDescriptor {
        let mut descriptor =
            Self::descriptor_of(self.columns(), dtype, activation, self.bias.is_some());
        descriptor.absmax = self.absmax;
        descriptor
    }

    /// The descriptor from the numbers alone: a residual of `dtype` folding updates of `activation`.
    pub fn descriptor_of(
        columns: u32,
        dtype: DType,
        activation: DType,
        update_bias: bool,
    ) -> residual::ScaledResidualRmsNormDescriptor {
        residual::ScaledResidualRmsNormDescriptor {
            dtype,
            columns,
            tile: row_tile(columns),
            update_bias,
            absmax: false,
            activation,
        }
    }

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
        let descriptor = self.descriptor(hidden.dtype, update.dtype);
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

        f.operate::<RmsNormOp>(&descriptor, args)
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


    /// Channels per row.
    pub fn columns(&self) -> u32 {
        self.weight.shape().numel() as u32
    }

    /// Fold `update` into `hidden` and normalize it into `out`.
    /// The descriptor one launch takes, for a walk without a device; see
    /// [`ScaledResidual::descriptor`].
    pub fn descriptor(
        &self,
        dtype: DType,
        activation: DType,
    ) -> residual::ScaledResidualLayerNormDescriptor {
        let mut descriptor = Self::descriptor_of(
            self.columns(),
            dtype,
            activation,
            self.update_bias.is_some(),
        );
        descriptor.absmax = self.absmax;
        descriptor
    }

    /// The descriptor from the numbers alone: a residual of `dtype` folding updates of `activation`.
    pub fn descriptor_of(
        columns: u32,
        dtype: DType,
        activation: DType,
        update_bias: bool,
    ) -> residual::ScaledResidualLayerNormDescriptor {
        residual::ScaledResidualLayerNormDescriptor {
            dtype,
            columns,
            tile: row_tile(columns),
            update_bias,
            absmax: false,
            activation,
        }
    }

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
        let descriptor = self.descriptor(hidden.dtype, update.dtype);
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

        f.operate::<LayerNormOp>(&descriptor, args)
    }
}

/// The diffusion solver's update: `target ← β · target + α · source`, in place.
///
/// The Euler step passes `α = σ − σ_next`, `β = 1`; the probability-flow start passes `β = σ₀`,
/// `α = 0`. Both coefficients are host values, so a captured graph stays valid, and the two
/// operands' element types differ on purpose — a BF16 trajectory stepped by an FP32 velocity
/// (specs/worker.md 9, 求解器; `ops::residual::ScaledAdd`).
pub struct ScaledAdd;

impl ScaledAdd {
    /// The descriptor one launch takes, for a walk without a device.
    ///
    /// The two operands' element types are the caller's — a BF16 trajectory stepped by an FP32
    /// velocity is the intended pairing — and the width is the operands', so both are arguments.
    pub fn descriptor(&self, target: DType, source: DType) -> residual::ScaledAddDescriptor {
        residual::ScaledAddDescriptor {
            target,
            source,
            block: BLOCK,
        }
    }

    /// `target ← β · target + α · source` over `[rows, columns]`, writing `target` in place.
    pub fn forward(
        &self,
        f: &Forward,
        target: TensorView,
        source: TensorView,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        if target.shape.rank() != 2 || target.shape != source.shape {
            return Err(Error::shape(
                "ScaledAdd",
                format!("{:?} with {:?}", target.shape.dims(), source.shape.dims()),
            ));
        }
        let descriptor = self.descriptor(target.dtype, source.dtype);
        f.operate::<residual::ScaledAdd>(
            &descriptor,
            residual::ScaledAddArgs {
                target,
                source,
                alpha,
                beta,
            },
        )
    }
}

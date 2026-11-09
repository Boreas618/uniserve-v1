//! Row-wise normalization: RMS normalization, its fused residual form, and layer normalization.
//!
//! Qwen3's decoder keeps a separate residual stream and fuses the residual add into the next norm:
//! the sum is formed in FP32 and rounded once to the hidden dtype, so
//! [`RmsNorm::forward_with_residual`] is one operator call and not an add followed by a norm.

use ops::norm::{
    self, AddRmsNorm, AddRmsNormDescriptor, LayerNorm as LayerNormOp, LayerNormDescriptor,
    RmsNorm as RmsNormOp, RmsNormDescriptor,
};
use ops::residual::{
    self, ScaledResidualLayerNorm as ScaledLayerNormOp, ScaledResidualRmsNorm as ScaledRmsNormOp,
};
use ops::{DType, TensorView};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::launch::{PDL, row_tile};
use crate::weights::{Scope, Weight};

/// RMS normalization with a per-channel scale.
pub struct RmsNorm {
    weight: Weight,
    eps: f32,
    columns: u32,
}

impl RmsNorm {
    /// The normalization of `weight`'s width with variance floor `eps`.
    pub fn new(weight: Weight, eps: f32) -> Result<Self> {
        let columns = weight.shape().numel();
        Ok(Self {
            weight,
            eps,
            columns: u32::try_from(columns)
                .map_err(|_| Error::shape("RmsNorm", format!("a width of {columns} elements")))?,
        })
    }

    /// The normalization module at `scope`: its `weight` of `width` channels, with variance floor
    /// `eps`.
    ///
    /// # Errors
    ///
    /// When the checkpoint lacks the weight or holds another width.
    pub fn load(scope: &mut Scope<'_, '_>, width: u64, eps: f32) -> Result<Self> {
        Self::new(scope.tensor("weight", &[width])?, eps)
    }

    /// The per-channel scale.
    pub fn weight(&self) -> TensorView {
        self.weight.view()
    }

    /// The descriptor [`Self::forward`] launches with, for an input of `input`.
    ///
    /// The row count is not in it: rows are a launch value, and a normalization is compiled per
    /// column count and element type; the tile follows the width.
    fn descriptor(&self, input: ops::DType) -> RmsNormDescriptor {
        RmsNormDescriptor {
            dtype: input,
            columns: self.columns,
            tile: row_tile(self.columns),
            pdl: PDL,
        }
    }

    /// The descriptor [`Self::forward_with_residual`] launches with, for an input of `input`.
    fn add_descriptor(&self, input: ops::DType) -> AddRmsNormDescriptor {
        AddRmsNormDescriptor {
            dtype: input,
            columns: self.columns,
            tile: row_tile(self.columns),
            pdl: PDL,
        }
    }

    /// `out = x · rsqrt(mean(x²) + eps) · weight`.
    pub fn forward(&self, f: &Forward, x: TensorView, out: TensorView) -> Result<()> {
        let columns = u64::from(self.columns);
        x.require_row("RmsNorm", x.dtype, columns)
            .map_err(Error::from)?;
        out.require_row("RmsNorm", out.dtype, columns)
            .map_err(Error::from)?;
        if out.rows() != x.rows() {
            return Err(Error::shape(
                "RmsNorm",
                format!("{} input rows against {} output rows", x.rows(), out.rows()),
            ));
        }
        let descriptor = self.descriptor(x.dtype);
        let args = norm::RmsNormArgs {
            x,
            weight: self.weight.view(),
            out,
            eps: self.eps,
        };

        f.operate::<RmsNormOp>(&descriptor, args)
    }

    /// `residual_out = x + residual` in FP32, `out = rms_norm(residual_out)`.
    ///
    /// `out` and `residual_out` may be the same buffer as `x` or as `residual`: the operator loads
    /// its whole row before storing.
    pub fn forward_with_residual(
        &self,
        f: &Forward,
        x: TensorView,
        residual: TensorView,
        residual_out: TensorView,
        out: TensorView,
    ) -> Result<()> {
        let columns = u64::from(self.columns);
        for (name, view) in [
            ("x", x),
            ("residual", residual),
            ("residual_out", residual_out),
        ] {
            view.require_row("RmsNorm", view.dtype, columns)
                .map_err(|error| Error::shape("RmsNorm", format!("{name}: {error}")))?;
            if view.rows() != x.rows() {
                return Err(Error::shape(
                    "RmsNorm",
                    format!("{name} has {} rows, {} expected", view.rows(), x.rows()),
                ));
            }
        }
        out.require_row("RmsNorm", out.dtype, columns)
            .map_err(Error::from)?;
        let descriptor = self.add_descriptor(x.dtype);
        let args = norm::AddRmsNormArgs {
            x,
            residual,
            weight: self.weight.view(),
            out,
            residual_out,
            eps: self.eps,
        };

        f.operate::<AddRmsNorm>(&descriptor, args)
    }
}

/// Layer normalization with a per-channel scale and an optional bias.
pub struct LayerNorm {
    weight: Weight,
    bias: Option<Weight>,
    eps: f32,
    columns: u32,
}

impl LayerNorm {
    /// The affine normalization module at `scope`: its `weight` and `bias` of `width` channels,
    /// with variance floor `eps`.
    ///
    /// # Errors
    ///
    /// When the checkpoint lacks either tensor or holds another width.
    pub fn load(scope: &mut Scope<'_, '_>, width: u64, eps: f32) -> Result<Self> {
        let weight = scope.tensor("weight", &[width])?;
        let bias = scope.tensor("bias", &[width])?;
        Self::new(weight, Some(bias), eps)
    }

    /// The normalization of `weight`'s width with variance floor `eps`.
    pub fn new(weight: Weight, bias: Option<Weight>, eps: f32) -> Result<Self> {
        let columns = weight.shape().numel();
        if let Some(bias) = bias
            && bias.shape().numel() != columns
        {
            return Err(Error::shape(
                "LayerNorm",
                format!(
                    "a bias of {} elements cannot scale {columns} channels",
                    bias.shape().numel()
                ),
            ));
        }
        Ok(Self {
            weight,
            bias,
            eps,
            columns: u32::try_from(columns)
                .map_err(|_| Error::shape("LayerNorm", format!("a width of {columns} elements")))?,
        })
    }

    /// The descriptor [`Self::forward`] launches with, for an input of `input`.
    ///
    /// The bias is in it, so a layer with and one without are two modules: the epilogue loads a
    /// second vector or it does not.
    fn descriptor(&self, input: ops::DType) -> LayerNormDescriptor {
        LayerNormDescriptor {
            dtype: input,
            columns: self.columns,
            tile: row_tile(self.columns),
            bias: self.bias.is_some(),
            pdl: PDL,
        }
    }

    /// `out = (x − mean(x)) · rsqrt(var(x) + eps) · weight + bias`.
    pub fn forward(&self, f: &Forward, x: TensorView, out: TensorView) -> Result<()> {
        let columns = u64::from(self.columns);
        x.require_row("LayerNorm", x.dtype, columns)
            .map_err(Error::from)?;
        out.require_row("LayerNorm", out.dtype, columns)
            .map_err(Error::from)?;
        if out.rows() != x.rows() {
            return Err(Error::shape(
                "LayerNorm",
                format!("{} input rows against {} output rows", x.rows(), out.rows()),
            ));
        }
        let descriptor = self.descriptor(x.dtype);
        let args = norm::LayerNormArgs {
            x,
            weight: self.weight.view(),
            bias: self.bias.as_ref().map(Weight::view),
            out,
            eps: self.eps,
        };

        f.operate::<LayerNormOp>(&descriptor, args)
    }
}

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

    /// The descriptor one launch without a magnitude takes: rows of `dtype`, normalized into
    /// `activation`.
    fn descriptor(&self, dtype: DType, activation: DType) -> residual::WeightedRmsNormDescriptor {
        let columns = self.columns();
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
        let mut descriptor = self.descriptor(x.dtype, out.dtype);
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

    /// The descriptor one launch takes: a residual of `dtype` folding updates of `activation`.
    fn descriptor(
        &self,
        dtype: DType,
        activation: DType,
    ) -> residual::ScaledResidualRmsNormDescriptor {
        let columns = self.columns();
        residual::ScaledResidualRmsNormDescriptor {
            dtype,
            columns,
            tile: row_tile(columns),
            update_bias: self.bias.is_some(),
            absmax: self.absmax,
            activation,
        }
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

        f.operate::<ScaledRmsNormOp>(&descriptor, args)
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

    /// The descriptor one launch takes: a residual of `dtype` folding updates of `activation`.
    fn descriptor(
        &self,
        dtype: DType,
        activation: DType,
    ) -> residual::ScaledResidualLayerNormDescriptor {
        let columns = self.columns();
        residual::ScaledResidualLayerNormDescriptor {
            dtype,
            columns,
            tile: row_tile(columns),
            update_bias: self.update_bias.is_some(),
            absmax: self.absmax,
            activation,
        }
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

        f.operate::<ScaledLayerNormOp>(&descriptor, args)
    }
}

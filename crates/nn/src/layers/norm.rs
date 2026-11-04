//! Row-wise normalization: RMS normalization, its fused residual form, and layer normalization.
//!
//! Qwen3's decoder keeps a separate residual stream and fuses the residual add into the next norm
//! (specs/worker.md 9, "RMSNorm 与 add-RMSNorm"; `uniserve/models/qwen3/transformer.py`): the sum is
//! formed in FP32 and rounded once to the hidden dtype, so [`RmsNorm::forward_with_residual`] is one
//! operator call and not an add followed by a norm.

use ops::norm::{
    self, AddRmsNorm, AddRmsNormDescriptor, LayerNorm as LayerNormOp, LayerNormDescriptor,
    RmsNorm as RmsNormOp, RmsNormDescriptor,
};
use ops::{Op, TensorView};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::ACCUMULATE;
use crate::layers::launch::{PDL, row_tile};
use crate::weights::Weight;

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

    /// The normalization's width.
    pub const fn columns(&self) -> u32 {
        self.columns
    }

    /// The variance floor.
    pub const fn eps(&self) -> f32 {
        self.eps
    }

    /// The per-channel scale.
    pub fn weight(&self) -> TensorView {
        self.weight.view()
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
        let descriptor = RmsNormDescriptor {
            dtype: x.dtype,
            columns: self.columns,
            tile: row_tile(self.columns),
            pdl: PDL,
        };
        let args = norm::RmsNormArgs {
            x,
            weight: self.weight.view(),
            out,
            eps: self.eps,
        };
        let kernel = f.kernels().op::<RmsNormOp>(&descriptor)?;
        RmsNormOp::launch(kernel, &descriptor, args, f.stream).map_err(Error::from)
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
        let descriptor = AddRmsNormDescriptor {
            dtype: x.dtype,
            columns: self.columns,
            tile: row_tile(self.columns),
            pdl: PDL,
        };
        let args = norm::AddRmsNormArgs {
            x,
            residual,
            weight: self.weight.view(),
            out,
            residual_out,
            eps: self.eps,
        };
        let kernel = f.kernels().op::<AddRmsNorm>(&descriptor)?;
        AddRmsNorm::launch(kernel, &descriptor, args, f.stream).map_err(Error::from)
    }

    /// The bound two implementations of this normalization may differ by, for shadow execution.
    pub fn tolerance(&self) -> ops::Tolerance {
        ops::Tolerance::rounding(u64::from(self.columns), ACCUMULATE, self.weight.dtype())
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
        let descriptor = LayerNormDescriptor {
            dtype: x.dtype,
            columns: self.columns,
            tile: row_tile(self.columns),
            bias: self.bias.is_some(),
            pdl: PDL,
        };
        let args = norm::LayerNormArgs {
            x,
            weight: self.weight.view(),
            bias: self.bias.as_ref().map(Weight::view),
            out,
            eps: self.eps,
        };
        let kernel = f.kernels().op::<LayerNormOp>(&descriptor)?;
        LayerNormOp::launch(kernel, &descriptor, args, f.stream).map_err(Error::from)
    }
}

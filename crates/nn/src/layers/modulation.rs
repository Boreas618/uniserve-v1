//! The diffusion transformer's modulation layer: an indexed RMS modulation and the gated residual.
//!
//! A diffusion block is conditioned on a timestep embedding: three vectors per conditioning row —
//! shift, scale and gate — are selected per activation row, the row is normalized, its affine image
//! is formed from that row's parameters, and a gated update is folded into the residual stream
//! (specs/worker.md 9, "调制 RMSNorm、门控残差"; `ops::modulation`). The layer owns the three
//! parameter tables and the epsilon; the activation, the row indices and the residual are the
//! caller's views.

use ops::descriptor::Float;
use ops::modulation::{self, ModulatedRmsNorm as ModulatedRmsNormOp};
use ops::{DType, Op, TensorView};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::launch::row_tile;
use crate::weights::Weight;

/// Elements between the parameter rows a `[parameters, width]` table selects.
fn row_stride(name: &'static str, table: &Weight, width: u64) -> Result<u64> {
    let view = table.view();
    if view.shape.rank() != 2 || view.shape.last() != width {
        return Err(Error::shape(
            name,
            format!(
                "a modulation table is [parameters, {width}], not {:?}",
                view.shape.dims()
            ),
        ));
    }
    Ok(view.strides.get(0))
}

/// The indexed RMS modulation of a diffusion block.
pub struct ModulatedRmsNorm {
    weight: Weight,
    shift: Weight,
    scale: Weight,
    gate: Option<Weight>,
    eps: f32,
    retain: bool,
}

impl ModulatedRmsNorm {
    /// The modulation of `weight`'s width: the per-row shift and scale tables, and the gate of the
    /// gated update when the block folds one.
    pub fn new(
        weight: Weight,
        shift: Weight,
        scale: Weight,
        gate: Option<Weight>,
        eps: f32,
    ) -> Result<Self> {
        if weight.shape().rank() != 1 {
            return Err(Error::shape(
                "ModulatedRmsNorm",
                format!(
                    "the learned scale is one vector, not {:?}",
                    weight.shape().dims()
                ),
            ));
        }
        let width = weight.shape().numel();
        for (name, table) in [("shift", &shift), ("scale", &scale)] {
            row_stride(name, table, width)?;
        }
        if let Some(gate) = &gate {
            row_stride("gate", gate, width)?;
        }
        Ok(Self {
            weight,
            shift,
            scale,
            gate,
            eps,
            retain: false,
        })
    }

    /// Copy the normalized source row out on every call (`[rows, width]`, in the activation's
    /// element type), for a block that needs it unmodulated.
    pub fn with_retain(mut self) -> Self {
        self.retain = true;
        self
    }

    /// Elements per activation row.
    pub fn width(&self) -> u64 {
        self.weight.shape().numel()
    }

    /// Whether the block folds a gated update before normalizing.
    pub fn has_update(&self) -> bool {
        self.gate.is_some()
    }

    /// Normalize and modulate `value`, and fold the gated update into it when the layer has one.
    ///
    /// `value` is `[rows, width]`; `update`, when present, is the block's output view of the same
    /// shape and receives `value + gate[i] · update`; `row_indices` is `[rows]` and selects the
    /// parameter row of each activation row; `out` is `[rows, width]` in the output element type
    /// (the activation's, or E4M3 with `output_scale` present); `retained`, when present, is
    /// `[rows, width]` and receives the normalized source.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        f: &Forward,
        value: TensorView,
        update: Option<TensorView>,
        row_indices: TensorView,
        out: TensorView,
        output_scale: Option<TensorView>,
        retained: Option<TensorView>,
    ) -> Result<()> {
        if update.is_some() != self.has_update() {
            return Err(Error::shape(
                "ModulatedRmsNorm",
                "the update view is present exactly when the layer has a gate".to_owned(),
            ));
        }
        if retained.is_some() != self.retain {
            return Err(Error::shape(
                "ModulatedRmsNorm",
                "the retained buffer is present exactly when the layer retains the source"
                    .to_owned(),
            ));
        }
        if (out.dtype == DType::F8E4M3) != output_scale.is_some() {
            return Err(Error::shape(
                "ModulatedRmsNorm",
                "an E4M3 output carries one dequantization scale per row, and only then".to_owned(),
            ));
        }
        let width = self.width();
        let width_u32 = u32::try_from(width)
            .map_err(|_| Error::shape("ModulatedRmsNorm", "a row wider than a u32".to_owned()))?;
        let shift = self.shift.view();
        let scale = self.scale.view();
        let descriptor = modulation::ModulatedRmsNormDescriptor {
            dtype: value.dtype,
            output: out.dtype,
            indices: row_indices.dtype,
            width: width_u32,
            block: row_tile(width_u32),
            epsilon: Float::from(self.eps),
            gate_stride: self
                .gate
                .as_ref()
                .map(|gate| gate.view().strides.get(0))
                .unwrap_or(0),
            shift_stride: shift.strides.get(0),
            scale_stride: scale.strides.get(0),
            update: update.is_some(),
            retain: retained.is_some(),
        };
        let args = modulation::Args {
            value,
            update,
            gate: self.gate.as_ref().map(Weight::view),
            weight: self.weight.view(),
            shift,
            scale,
            row_indices,
            out,
            output_scale,
            retain: retained,
        };
        let kernel = f.kernels().op::<ModulatedRmsNormOp>(&descriptor)?;
        ModulatedRmsNormOp::launch(kernel, &descriptor, args, f.stream).map_err(Error::from)
    }
}

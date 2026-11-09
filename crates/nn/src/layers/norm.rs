//! Normalization: RMS and layer normalization over rows, each with the elementwise stages fused
//! around its row statistics, and group normalization over a video frame's channel groups.
//!
//! A residual stream is normalized where it is updated, so the add and the normalization are one
//! launch that reads the row once. The normalization's [`Epilogue`] states what is fused:
//!
//! ```text
//! None             residual? s = x + residual                 out = norm(s or x)
//! ScaledResidual   s = residual + (x + bias) · scale           out = norm(s)
//! Modulated        s = residual + gate[row] · x  (optional)    out = norm(s or x) · (1 + scale[row]) + shift[row]
//! ```
//!
//! The residual stream is a call operand ([`ResidualStream`]): a decoder's first layer normalizes its
//! embedding alone and every later layer adds into the stream first, with the same module, as the
//! reference's `forward(x, residual=None)` does. Every sum is formed in FP32; the stream stores it
//! rounded once, and the normalization reads the unrounded sum.
//!
//! The forms are distinct operators because they read distinct operands, and the layer launches the
//! one its epilogue and call name: `rms_norm`, `add_rms_norm`, `weighted_rms_norm` (an RMS
//! normalization that narrows into a smaller element type), `scaled_residual_rms_norm`,
//! `modulated_rms_norm`; `layer_norm` and `scaled_residual_layer_norm`.
//!
//! [`GroupNorm`] normalizes a frame's channel groups for a causal video convolution, so its fused
//! stage is that convolution's pre-activation: SiLU and the causal pad (`frame_moments`, then
//! `frame_norm_pad`).

use ops::TensorView;
use ops::descriptor::Float;
use ops::frame::{self, FrameMoments, FrameNormPad};
use ops::modulation::{self, ModulatedRmsNorm};
use ops::norm::{self, AddRmsNorm, LayerNorm as LayerNormOp, RmsNorm as RmsNormOp};
use ops::residual::{self, ScaledResidualLayerNorm, ScaledResidualRmsNorm, WeightedRmsNorm};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::CausalPad;
use crate::layers::launch::{PDL, row_tile};
use crate::weights::{Scope, Weight};

/// The residual stream a normalization adds its input into before it normalizes.
///
/// `input` is read and `output` receives the FP32 sum rounded once into its element type. They are
/// one buffer for a stream updated in place; a decoder's first add reads the embedding and writes
/// the stream's own region.
#[derive(Clone, Copy, Debug)]
pub struct ResidualStream {
    /// The stream before the add.
    pub input: TensorView,
    /// The stream after the add.
    pub output: TensorView,
}

impl ResidualStream {
    /// A stream updated in place.
    pub fn in_place(inout: TensorView) -> Self {
        Self {
            input: inout,
            output: inout,
        }
    }

    fn in_place_only(&self, owner: &'static str) -> Result<TensorView> {
        if !same(&self.input, &self.output) {
            return Err(Error::shape(
                owner,
                "this epilogue updates its residual stream in place".to_owned(),
            ));
        }
        Ok(self.input)
    }
}

/// The per-row modulation of a diffusion block: three tables a conditioning row selects from.
///
/// The tables are `[parameters, width]` with unit channel stride (rows of the step's gathered
/// modulation); `rows` is the call's `[rows]` index of each activation row's parameter row. The
/// modulation is built for the call that reads it, because the tables are the step's.
#[derive(Clone, Copy, Debug)]
pub struct Modulation {
    /// The per-row shift added after the scale.
    pub shift: Weight,
    /// The per-row scale offset: the normalized row is multiplied by `1 + scale`.
    pub scale: Weight,
    /// The gate of the update folded into the stream first; present exactly when the call passes a
    /// [`ResidualStream`].
    pub gate: Option<Weight>,
    /// `[rows]`: the parameter row of each activation row.
    pub rows: TensorView,
    /// `[rows, width]` in the output type: receives the stream's sum as well, for a block that
    /// reads it again after `x` has been overwritten.
    pub retained: Option<TensorView>,
}

/// The stages a normalization fuses around its statistics; see the module table.
#[derive(Clone, Copy, Debug)]
#[allow(
    clippy::large_enum_variant,
    reason = "one epilogue per normalization, never held in bulk; boxing would allocate per call"
)]
pub enum Epilogue {
    /// The normalization alone, or after a plain add when the call passes a [`ResidualStream`].
    None,
    /// A per-channel scaled update `(x + bias) · scale` folded into the stream, in place.
    ScaledResidual {
        /// `[columns]`: the update's per-channel scale (layer scale).
        scale: Weight,
        /// `[columns]`: the update's bias, when the block's projection leaves it out.
        bias: Option<Weight>,
    },
    /// The per-row modulation of the normalized row (RMS normalization only).
    Modulated(Modulation),
}

/// Whether two views name the same elements.
fn same(a: &TensorView, b: &TensorView) -> bool {
    a.ptr == b.ptr && a.dtype == b.dtype && a.shape == b.shape && a.strides == b.strides
}

/// The width of a `[columns]` vector.
fn columns(owner: &'static str, weight: &Weight) -> Result<u32> {
    if weight.shape().rank() != 1 {
        return Err(Error::shape(
            owner,
            format!(
                "a per-channel parameter is one vector, not {:?}",
                weight.shape().dims()
            ),
        ));
    }
    u32::try_from(weight.shape().numel())
        .map_err(|_| Error::shape(owner, "more channels than a descriptor holds".to_owned()))
}

/// Checks an epilogue's parameters against the normalization's `width`.
fn check_epilogue(owner: &'static str, epilogue: &Epilogue, width: u32) -> Result<()> {
    let same_width = |name: &str, weight: &Weight| -> Result<()> {
        if columns(owner, weight)? != width {
            return Err(Error::shape(
                owner,
                format!("the {name} and the normalization's weight differ in channels"),
            ));
        }
        Ok(())
    };
    match epilogue {
        Epilogue::None => Ok(()),
        Epilogue::ScaledResidual { scale, bias } => {
            same_width("residual's scale", scale)?;
            if let Some(bias) = bias {
                same_width("update's bias", bias)?;
            }
            Ok(())
        }
        Epilogue::Modulated(modulation) => {
            let tables = [
                Some(("shift", &modulation.shift)),
                Some(("scale", &modulation.scale)),
                modulation.gate.as_ref().map(|gate| ("gate", gate)),
            ];
            for (name, table) in tables.into_iter().flatten() {
                let shape = table.shape();
                if shape.rank() != 2 || shape.last() != u64::from(width) {
                    return Err(Error::shape(
                        owner,
                        format!(
                            "a {name} table is [parameters, {width}], not {:?}",
                            shape.dims()
                        ),
                    ));
                }
            }
            Ok(())
        }
    }
}

/// Checks that `view` is `[rows, width]` rows of `x`'s count.
fn require_rows(
    owner: &'static str,
    name: &str,
    view: &TensorView,
    x: &TensorView,
    width: u32,
) -> Result<()> {
    view.require_row(owner, view.dtype, u64::from(width))
        .map_err(|error| Error::shape(owner, format!("{name}: {error}")))?;
    if view.rows() != x.rows() {
        return Err(Error::shape(
            owner,
            format!("{name} has {} rows, {} expected", view.rows(), x.rows()),
        ));
    }
    Ok(())
}

/// RMS normalization with a per-channel scale: `norm(s) = s · rsqrt(mean(s²) + eps) · weight`.
pub struct RmsNorm {
    weight: Weight,
    eps: f32,
    columns: u32,
    epilogue: Epilogue,
}

impl RmsNorm {
    /// The normalization of `weight`'s width with variance floor `eps` and `epilogue`.
    ///
    /// # Errors
    ///
    /// A weight that is not one vector, or epilogue parameters of another width.
    pub fn new(weight: Weight, eps: f32, epilogue: Epilogue) -> Result<Self> {
        let columns = columns("RmsNorm", &weight)?;
        check_epilogue("RmsNorm", &epilogue, columns)?;
        Ok(Self {
            weight,
            eps,
            columns,
            epilogue,
        })
    }

    /// The normalization module at `scope`: its `weight` of `width` channels, with variance floor
    /// `eps` and no epilogue.
    ///
    /// # Errors
    ///
    /// When the checkpoint lacks the weight or holds another width.
    pub fn load(scope: &mut Scope<'_, '_>, width: u32, eps: f32) -> Result<Self> {
        Self::new(
            scope.tensor("weight", &[u64::from(width)])?,
            eps,
            Epilogue::None,
        )
    }

    /// The per-channel scale.
    pub fn weight(&self) -> TensorView {
        self.weight.view()
    }

    /// Normalize `x`, `[rows, columns]`, into `out`, after adding it into `residual` when given.
    ///
    /// Under [`Epilogue::None`] the add is plain and `out` may alias `x` or the stream: the operator
    /// loads its whole row before storing. Under [`Epilogue::ScaledResidual`] the stream is required
    /// and updated in place. Under [`Epilogue::Modulated`] a stream is passed exactly when the
    /// modulation has a gate, and its output is `x` itself, which receives the sum. An `out` of a
    /// narrower element type than `x` is written by the narrowing RMS operator; every other form
    /// writes `x`'s element type.
    ///
    /// # Errors
    ///
    /// Shapes or element types the epilogue's operator does not take, a stream the epilogue does
    /// not read, or the operator's refusal.
    pub fn forward(
        &self,
        f: &Forward,
        x: TensorView,
        residual: Option<ResidualStream>,
        out: TensorView,
    ) -> Result<()> {
        const OWNER: &str = "RmsNorm";
        let width = self.columns;
        require_rows(OWNER, "x", &x, &x, width)?;
        require_rows(OWNER, "out", &out, &x, width)?;
        if let Some(stream) = &residual {
            require_rows(OWNER, "the residual stream", &stream.input, &x, width)?;
            require_rows(OWNER, "the residual stream", &stream.output, &x, width)?;
        }
        let tile = row_tile(width);

        match (&self.epilogue, residual) {
            (Epilogue::None, None) if out.dtype == x.dtype => f.operate::<RmsNormOp>(
                &norm::RmsNormDescriptor {
                    dtype: x.dtype,
                    columns: width,
                    tile,
                    pdl: PDL,
                },
                norm::RmsNormArgs {
                    x,
                    weight: self.weight.view(),
                    out,
                    eps: self.eps,
                },
            ),
            (Epilogue::None, None) => f.operate::<WeightedRmsNorm>(
                &residual::WeightedRmsNormDescriptor {
                    dtype: x.dtype,
                    columns: width,
                    tile,
                    absmax: false,
                    activation: out.dtype,
                },
                residual::WeightedRmsNormArgs {
                    x,
                    weight: self.weight.view(),
                    out,
                    eps: self.eps,
                    partials: None,
                },
            ),
            (Epilogue::None, Some(stream)) => f.operate::<AddRmsNorm>(
                &norm::AddRmsNormDescriptor {
                    dtype: x.dtype,
                    columns: width,
                    tile,
                    pdl: PDL,
                },
                norm::AddRmsNormArgs {
                    x,
                    residual: stream.input,
                    weight: self.weight.view(),
                    out,
                    residual_out: stream.output,
                    eps: self.eps,
                },
            ),
            (Epilogue::ScaledResidual { .. }, None) => Err(Error::shape(
                OWNER,
                "a scaled residual normalizes the stream it updates, and none was passed"
                    .to_owned(),
            )),
            (Epilogue::ScaledResidual { scale, bias }, Some(stream)) => {
                let hidden = stream.in_place_only(OWNER)?;
                f.operate::<ScaledResidualRmsNorm>(
                    &residual::ScaledResidualRmsNormDescriptor {
                        dtype: hidden.dtype,
                        columns: width,
                        tile,
                        update_bias: bias.is_some(),
                        absmax: false,
                        activation: x.dtype,
                    },
                    residual::ScaledResidualRmsNormArgs {
                        hidden,
                        update: x,
                        update_bias: bias.as_ref().map(Weight::view),
                        scale: scale.view(),
                        weight: self.weight.view(),
                        out,
                        eps: self.eps,
                        partials: None,
                    },
                )
            }
            (Epilogue::Modulated(modulation), stream) => {
                self.modulated(f, modulation, x, stream, out)
            }
        }
    }

    /// Normalize `x` into a fresh `[rows, columns]` buffer of `x`'s element type reserved from the
    /// arena, after adding it into `residual` when given; see [`Self::forward`].
    ///
    /// # Errors
    ///
    /// An arena too small for the buffer, or [`Self::forward`]'s refusals.
    pub fn apply(
        &self,
        f: &Forward,
        x: TensorView,
        residual: Option<ResidualStream>,
    ) -> Result<TensorView> {
        let out = f.arena().tensor(x.dtype, x.shape)?;
        self.forward(f, x, residual, out)?;
        Ok(out)
    }

    /// The `modulated_rms_norm` launch: the optional gated update of the stream, the normalization
    /// and the per-row modulation.
    fn modulated(
        &self,
        f: &Forward,
        modulation: &Modulation,
        x: TensorView,
        stream: Option<ResidualStream>,
        out: TensorView,
    ) -> Result<()> {
        const OWNER: &str = "RmsNorm";
        // The operator reads the stream from `value` and writes the gated sum over its update
        // operand, so the stream's output is the update itself.
        let (value, update) = match (stream, modulation.gate.is_some()) {
            (None, false) => (x, None),
            (Some(stream), true) => {
                if !same(&stream.output, &x) {
                    return Err(Error::shape(
                        OWNER,
                        "a gated modulation writes the stream's sum over its update".to_owned(),
                    ));
                }
                (stream.input, Some(x))
            }
            _ => {
                return Err(Error::shape(
                    OWNER,
                    "a residual stream is passed exactly when the modulation has a gate".to_owned(),
                ));
            }
        };
        if let Some(retained) = &modulation.retained {
            require_rows(OWNER, "the retained rows", retained, &x, self.columns)?;
        }
        let shift = modulation.shift.view();
        let scale = modulation.scale.view();
        let descriptor = modulation::ModulatedRmsNormDescriptor {
            dtype: value.dtype,
            output: out.dtype,
            indices: modulation.rows.dtype,
            width: self.columns,
            block: row_tile(self.columns),
            epsilon: Float::from(self.eps),
            gate_stride: modulation
                .gate
                .as_ref()
                .map(|gate| gate.view().strides.get(0))
                .unwrap_or(0),
            shift_stride: shift.strides.get(0),
            scale_stride: scale.strides.get(0),
            update: update.is_some(),
            retain: modulation.retained.is_some(),
        };
        let args = modulation::Args {
            value,
            update,
            gate: modulation.gate.as_ref().map(Weight::view),
            weight: self.weight.view(),
            shift,
            scale,
            row_indices: modulation.rows,
            out,
            output_scale: None,
            retain: modulation.retained,
        };
        f.operate::<ModulatedRmsNorm>(&descriptor, args)
    }
}

/// Layer normalization with a per-channel scale and an optional bias:
/// `norm(s) = (s − mean(s)) · rsqrt(var(s) + eps) · weight + bias`.
pub struct LayerNorm {
    weight: Weight,
    bias: Option<Weight>,
    eps: f32,
    columns: u32,
    epilogue: Epilogue,
}

impl LayerNorm {
    /// The affine normalization module at `scope`: its `weight` and `bias` of `width` channels,
    /// with variance floor `eps` and no epilogue.
    ///
    /// # Errors
    ///
    /// When the checkpoint lacks either tensor or holds another width.
    pub fn load(scope: &mut Scope<'_, '_>, width: u32, eps: f32) -> Result<Self> {
        let width = u64::from(width);
        let weight = scope.tensor("weight", &[width])?;
        let bias = scope.tensor("bias", &[width])?;
        Self::new(weight, Some(bias), eps, Epilogue::None)
    }

    /// The normalization of `weight`'s width with variance floor `eps` and `epilogue`.
    ///
    /// # Errors
    ///
    /// A bias or epilogue parameters of another width, a modulated epilogue (an RMS form only), or
    /// a scaled residual over a normalization without a bias (its operator always adds one).
    pub fn new(weight: Weight, bias: Option<Weight>, eps: f32, epilogue: Epilogue) -> Result<Self> {
        const OWNER: &str = "LayerNorm";
        let columns = columns(OWNER, &weight)?;
        if let Some(bias) = &bias
            && self::columns(OWNER, bias)? != columns
        {
            return Err(Error::shape(
                OWNER,
                format!(
                    "a bias of {} elements cannot shift {columns} channels",
                    bias.shape().numel()
                ),
            ));
        }
        match epilogue {
            Epilogue::Modulated(_) => {
                return Err(Error::shape(
                    OWNER,
                    "a modulated epilogue is an RMS normalization's".to_owned(),
                ));
            }
            Epilogue::ScaledResidual { .. } if bias.is_none() => {
                return Err(Error::shape(
                    OWNER,
                    "a scaled residual layer normalization adds its bias".to_owned(),
                ));
            }
            _ => {}
        }
        check_epilogue(OWNER, &epilogue, columns)?;
        Ok(Self {
            weight,
            bias,
            eps,
            columns,
            epilogue,
        })
    }

    /// Normalize `x`, `[rows, columns]`, into `out`, after folding it into `residual` under a
    /// scaled-residual epilogue (which updates the stream in place).
    ///
    /// # Errors
    ///
    /// Shapes the operator does not take, a stream passed without a scaled-residual epilogue or
    /// missing under one, or the operator's refusal.
    pub fn forward(
        &self,
        f: &Forward,
        x: TensorView,
        residual: Option<ResidualStream>,
        out: TensorView,
    ) -> Result<()> {
        const OWNER: &str = "LayerNorm";
        let width = self.columns;
        require_rows(OWNER, "x", &x, &x, width)?;
        require_rows(OWNER, "out", &out, &x, width)?;
        let tile = row_tile(width);

        match (&self.epilogue, residual) {
            (Epilogue::None, None) => f.operate::<LayerNormOp>(
                // The bias is in the descriptor, so a layer with and one without are two modules:
                // the epilogue loads a second vector or it does not.
                &norm::LayerNormDescriptor {
                    dtype: x.dtype,
                    columns: width,
                    tile,
                    bias: self.bias.is_some(),
                    pdl: PDL,
                },
                norm::LayerNormArgs {
                    x,
                    weight: self.weight.view(),
                    bias: self.bias.as_ref().map(Weight::view),
                    out,
                    eps: self.eps,
                },
            ),
            (Epilogue::ScaledResidual { scale, bias }, Some(stream)) => {
                let hidden = stream.in_place_only(OWNER)?;
                require_rows(OWNER, "the residual stream", &hidden, &x, width)?;
                let shift = self.bias.as_ref().ok_or_else(|| {
                    Error::shape(
                        OWNER,
                        "a scaled residual layer normalization adds its bias".to_owned(),
                    )
                })?;
                f.operate::<ScaledResidualLayerNorm>(
                    &residual::ScaledResidualLayerNormDescriptor {
                        dtype: hidden.dtype,
                        columns: width,
                        tile,
                        update_bias: bias.is_some(),
                        absmax: false,
                        activation: x.dtype,
                    },
                    residual::ScaledResidualLayerNormArgs {
                        hidden,
                        update: x,
                        update_bias: bias.as_ref().map(Weight::view),
                        scale: scale.view(),
                        weight: self.weight.view(),
                        bias: shift.view(),
                        out,
                        eps: self.eps,
                        partials: None,
                    },
                )
            }
            _ => Err(Error::shape(
                OWNER,
                "a layer normalization takes a residual stream exactly under a scaled residual"
                    .to_owned(),
            )),
        }
    }
}

/// Group normalization of a video frame's channel groups, with the frame's SiLU activation and its
/// causal pad fused into the store.
///
/// ```text
/// mean, rstd = group statistics of each frame          frame_moments (FP32, Welford)
/// out        = pad(silu(norm(values) · weight + bias))  frame_norm_pad
/// ```
///
/// The epilogue is the one a causal video convolution's pre-activation needs, and the two launches
/// read the frame twice instead of three times: the statistics, then one pass that normalizes,
/// activates and pads.
pub struct GroupNorm {
    groups: u32,
    eps: f32,
    weight: Option<Weight>,
    bias: Option<Weight>,
    pad: CausalPad,
}

impl GroupNorm {
    /// The normalization of `groups` channel groups with variance floor `eps`, the learned affine
    /// pair when the checkpoint has one, and the causal `pad` of the frame it writes.
    ///
    /// # Errors
    ///
    /// A weight without a bias or a bias without a weight, or no groups.
    pub fn new(
        groups: u32,
        eps: f32,
        weight: Option<Weight>,
        bias: Option<Weight>,
        pad: CausalPad,
    ) -> Result<Self> {
        if weight.is_some() != bias.is_some() {
            return Err(Error::shape(
                "GroupNorm",
                "the affine parameters come in a weight and a bias, or not at all".to_owned(),
            ));
        }
        if groups == 0 {
            return Err(Error::shape(
                "GroupNorm",
                "a normalization of no groups has no statistics".to_owned(),
            ));
        }
        Ok(Self {
            groups,
            eps,
            weight,
            bias,
            pad,
        })
    }

    /// Normalize `values`, activate them and pad them into `out`.
    ///
    /// `values` is `[batch, channels, frames, height, width]` FP32 with each frame's pixels
    /// contiguous; `mean` and `rstd` are the `[batch · frames, groups]` FP32 statistics buffers the
    /// call fills; `out` is the padded frame in the activation's element type, in either storage
    /// order — its strides state which, and they select the compiled variant.
    ///
    /// # Errors
    ///
    /// An output of another rank or channel count, one smaller than the source, or the operators'
    /// refusals.
    pub fn forward(
        &self,
        f: &Forward,
        values: TensorView,
        mean: TensorView,
        rstd: TensorView,
        out: TensorView,
    ) -> Result<()> {
        const OWNER: &str = "GroupNorm";
        if out.shape.rank() != 5 || values.shape.rank() != 5 {
            return Err(Error::shape(
                OWNER,
                format!(
                    "a frame is rank 5, not {:?} into {:?}",
                    values.shape.dims(),
                    out.shape.dims()
                ),
            ));
        }
        // The axes are logical in either storage order — `[batch, channels, frames, height, width]`
        // — and the order lives in the strides.
        let channels = values.shape.dim(1);
        if out.shape.dim(1) != channels {
            return Err(Error::shape(
                OWNER,
                format!(
                    "an output of {} channels for a source of {channels}",
                    out.shape.dim(1)
                ),
            ));
        }
        if (2..5).any(|axis| out.shape.dim(axis) < values.shape.dim(axis)) {
            return Err(Error::shape(
                OWNER,
                "the output's frames, rows and columns cannot be smaller than the source's"
                    .to_owned(),
            ));
        }
        // Both launches take the same reduction: `channels_per_group · pixels`, padded to the power
        // of two the compiled program is bounded by.
        let pixels = values.shape.dim(3) * values.shape.dim(4);
        let reduction = frame::frame_reduction_class(channels / u64::from(self.groups), pixels);

        f.operate::<FrameMoments>(
            &frame::FrameMomentsDescriptor {
                dtype: values.dtype,
                reduction,
                epsilon: Float::from(self.eps),
            },
            frame::FrameMomentsArgs {
                values,
                mean,
                rstd,
                groups: u64::from(self.groups),
            },
        )?;
        f.operate::<FrameNormPad>(
            &frame::FrameNormPadDescriptor {
                dtype: values.dtype,
                reduction,
                // The variant follows the output's storage, which the view states; the operator
                // checks the two agree before it launches.
                channels_last: frame::is_channels_last(&out),
                centered: self.weight.is_none(),
                single_pixel: out.shape.dim(3) * out.shape.dim(4) == 1,
            },
            frame::FrameNormPadArgs {
                values,
                mean,
                rstd,
                weight: self.weight.as_ref().map(Weight::view),
                bias: self.bias.as_ref().map(Weight::view),
                out,
                groups: u64::from(self.groups),
                top: u64::from(self.pad.top),
                left: u64::from(self.pad.left),
                front: u64::from(self.pad.front),
                reflect: self.pad.reflect(),
            },
        )
    }
}

//! The DAC audio encoder: a waveform to latent features through strided widening stages of
//! Snake-activated residual units.
//!
//! ```text
//! block.0                   conv, 1 → d_model channels, kernel 7, padding 3
//! block.<s + 1>.block       per stage of stride r, at the incoming width w:
//!     .0, .1, .2            residual units at dilations 1, 3, 9:
//!                               x += conv1x1(snake(conv7_dilated(snake(x))))
//!     .3                    snake
//!     .4                    conv, w → 2w channels, kernel 2r, stride r, padding ceil(r / 2)
//! block.<S + 1>             snake
//! block.<S + 2>             conv, → d_latent channels, kernel 3, padding 1
//! ```
//!
//! Every convolution is weight-normalized in the checkpoint and folded once at load, and every
//! Snake is `x + sin²(α·x) / (α + 1e-9)` with the checkpoint's raw per-channel `α`. A track of `L`
//! samples becomes `L / hop` frames, `hop` the product of the strides: the caller right-pads a
//! track to a whole number of hops, as the reference does before it convolves.

use ops::{Shape, Operand};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::{Bias, Conv1d, ConvSpec, Residual, Snake};
use crate::weights::{Scope, Weight};

/// The residual units' dilations, in the order the checkpoint names the units.
const DILATIONS: [u32; 3] = [1, 3, 9];

/// A DAC encoder's architecture, as its config states it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DacSpec {
    /// Channels the input convolution writes (`d_model`); each stage doubles them.
    pub channels: u32,
    /// Each stage's stride.
    pub strides: Vec<u32>,
    /// Channels the output convolution writes (`d_latent`).
    pub latent_channels: u32,
}

/// A Snake activation and its per-channel `α`.
struct Activation {
    snake: Snake,
    alpha: Weight,
}

impl Activation {
    /// The Snake module at `scope` over `channels`: its `alpha`.
    fn load(scope: &mut Scope<'_, '_>, channels: u32) -> Result<Self> {
        Ok(Self {
            snake: Snake::new(u64::from(channels))?,
            alpha: scope.tensor("alpha", &[u64::from(channels)])?,
        })
    }

    fn forward(&self, f: &Forward, x: Operand) -> Result<Operand> {
        self.snake.forward(f, x, self.alpha.view())
    }
}

/// One residual unit: Snake, dilated convolution, Snake, pointwise convolution, added back.
struct Unit {
    activations: [Activation; 2],
    convs: [Conv1d; 2],
}

/// One strided stage: three residual units at the incoming width, a Snake and the widening
/// convolution.
struct Stage {
    stride: u32,
    units: Vec<Unit>,
    activation: Activation,
    widen: Conv1d,
}

/// The DAC encoder from `block.0` to its output convolution.
pub struct DacEncoder {
    input: Conv1d,
    stages: Vec<Stage>,
    tail: Activation,
    output: Conv1d,
}

/// A weight-normalized 1-d convolution spec with torch's other defaults.
fn normalized(from: u32, to: u32, kernel: u32, padding: u32, dilation: u32) -> ConvSpec<1> {
    ConvSpec {
        padding: [padding],
        dilation: [dilation],
        bias: Bias::Present,
        weight_norm: true,
        ..ConvSpec::new(from, to, [kernel])
    }
}

impl DacEncoder {
    /// The encoder at `scope` (the reference's `encoder.block`).
    ///
    /// # Errors
    ///
    /// No stage, a zero stride, widths that overflow, or a module the checkpoint lacks.
    pub fn load(scope: &mut Scope<'_, '_>, spec: &DacSpec) -> Result<Self> {
        if spec.strides.is_empty() || spec.strides.contains(&0) || spec.channels == 0 {
            return Err(Error::shape(
                "DacEncoder",
                format!("{spec:?} has no stage, a zero stride or no width"),
            ));
        }
        let mut width = spec.channels;
        let input = Conv1d::load(&mut scope.scope("0"), normalized(1, width, 7, 3, 1))?;
        let mut stages = Vec::with_capacity(spec.strides.len());
        for (index, &stride) in spec.strides.iter().enumerate() {
            let inner = width;
            width = width.checked_mul(2).ok_or_else(|| {
                Error::shape("DacEncoder", "the stage widths overflow".to_owned())
            })?;
            let mut stage = scope.scope(format!("{}.block", index + 1));
            let units = DILATIONS
                .iter()
                .enumerate()
                .map(|(unit, &dilation)| {
                    let mut module = stage.scope(format!("{unit}.block"));
                    Ok(Unit {
                        convs: [
                            Conv1d::load(
                                &mut module.scope("1"),
                                normalized(inner, inner, 7, 3 * dilation, dilation),
                            )?,
                            Conv1d::load(
                                &mut module.scope("3"),
                                normalized(inner, inner, 1, 0, 1),
                            )?,
                        ],
                        activations: [
                            Activation::load(&mut module.scope("0"), inner)?,
                            Activation::load(&mut module.scope("2"), inner)?,
                        ],
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let activation = Activation::load(&mut stage.scope("3"), inner)?;
            let widen = Conv1d::load(
                &mut stage.scope("4"),
                ConvSpec {
                    stride: [stride],
                    ..normalized(inner, width, 2 * stride, stride.div_ceil(2), 1)
                },
            )?;
            stages.push(Stage {
                stride,
                units,
                activation,
                widen,
            });
        }
        let last = spec.strides.len();
        let tail = Activation::load(&mut scope.scope(last + 1), width)?;
        let output = Conv1d::load(
            &mut scope.scope(last + 2),
            normalized(width, spec.latent_channels, 3, 1, 1),
        )?;
        Ok(Self {
            input,
            stages,
            tail,
            output,
        })
    }

    /// The samples one output frame covers: the product of the strides.
    pub fn hop(&self) -> u64 {
        self.stages
            .iter()
            .map(|stage| u64::from(stage.stride))
            .product()
    }

    /// `samples` `[batch, 1, length]` encoded: a fresh `[batch, latent_channels, length / hop]`
    /// tensor of their element type.
    ///
    /// `length` is a whole number of hops. Each stage's stream is reserved before the stage and
    /// every unit's intermediates are handed back when the unit's residual add is enqueued.
    ///
    /// # Errors
    ///
    /// A length that is not a whole number of hops, samples of another shape, or an operator's
    /// refusal.
    pub fn forward(&self, f: &Forward, samples: Operand) -> Result<Operand> {
        let length = samples.shape.dim(2);
        let hop = self.hop();
        if samples.shape.rank() != 3 || samples.shape.dim(1) != 1 || !length.is_multiple_of(hop) {
            return Err(Error::shape(
                "DacEncoder",
                format!(
                    "{:?}: a [batch, 1, samples] track of whole {hop}-sample hops",
                    samples.shape.dims()
                ),
            ));
        }
        let mut hidden = self.input.forward(f, samples)?;

        // A stage's units add into the residual stream `hidden` in place; their own activations
        // and convolutions live only until the add is enqueued.
        for stage in &self.stages {
            for unit in &stage.units {
                f.scoped(|f| {
                    let first =
                        unit.convs[0].forward(f, unit.activations[0].forward(f, hidden)?)?;
                    let second =
                        unit.convs[1].forward(f, unit.activations[1].forward(f, first)?)?;
                    Residual::default().forward(f, matrix(hidden), matrix(second), None)?;
                    Ok::<_, Error>(())
                })?;
            }
            // The widened stream outlives the stage's activation, so it is reserved first.
            let next = stage.widen.reserve(f, hidden)?;
            f.scoped(|f| {
                let activated = stage.activation.forward(f, hidden)?;
                stage.widen.launch(f, activated, next)
            })?;
            hidden = next;
        }
        let out = self.output.reserve(f, hidden)?;
        f.scoped(|f| {
            let activated = self.tail.forward(f, hidden)?;
            self.output.launch(f, activated, out)
        })?;
        Ok(out)
    }
}

/// A contiguous `[batch, rows, columns]` view as `[batch · rows, columns]`.
fn matrix(view: Operand) -> Operand {
    view.reshape(Shape::new(&[
        view.shape.dim(0) * view.shape.dim(1),
        view.shape.dim(2),
    ]))
}

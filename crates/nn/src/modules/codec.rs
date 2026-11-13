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
//! Snake is `x + sin²(α·x) / (α + 1e-9)` with the checkpoint's raw per-channel `α`
//! ([`functional::snake`]). A track of `L` samples becomes `L / hop` frames, `hop` the product
//! of the strides: the caller right-pads a track to a whole number of hops, as the reference
//! does before it convolves.

use tensor::{Module, Tensor};

use crate::error::{Error, Result};
use crate::functional;
use crate::modules::conv::{Conv1d, ConvConfig};
use crate::var_builder::VarBuilder;

/// The residual units' dilations, in the order the checkpoint names the units.
const DILATIONS: [u32; 3] = [1, 3, 9];

/// A DAC encoder's architecture, as its checkpoint's configuration states it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DacEncoderConfig {
    /// Channels the input convolution writes (`d_model`); each stage doubles them.
    pub channels: u32,
    /// Each stage's stride.
    pub strides: Vec<u32>,
    /// Channels the output convolution writes (`d_latent`).
    pub latent_channels: u32,
}

/// One residual unit: Snake, dilated convolution, Snake, pointwise convolution, added back.
/// Each Snake is its per-channel `α`.
struct Unit {
    alphas: [Tensor; 2],
    convs: [Conv1d; 2],
}

/// One strided stage: three residual units at the incoming width, a Snake and the widening
/// convolution.
struct Stage {
    stride: u32,
    units: Vec<Unit>,
    alpha: Tensor,
    widen: Conv1d,
}

/// The DAC encoder from `block.0` to its output convolution.
pub struct DacEncoder {
    input: Conv1d,
    stages: Vec<Stage>,
    /// The `α` of the Snake before the output convolution.
    alpha: Tensor,
    output: Conv1d,
}

/// A weight-normalized 1-d convolution with torch's other defaults.
fn normalized(from: u32, to: u32, kernel: u32, padding: u32, dilation: u32) -> ConvConfig<1> {
    ConvConfig {
        padding: [padding],
        dilation: [dilation],
        weight_norm: true,
        ..ConvConfig::new(from, to, [kernel])
    }
}

/// The per-channel `α` of the Snake module at `vb`.
fn alpha(vb: VarBuilder<'_, '_>, channels: u32) -> Result<Tensor> {
    vb.get(channels as usize, "alpha")
}

impl DacEncoder {
    /// The encoder at `vb` (the reference's `encoder.block`).
    ///
    /// # Errors
    ///
    /// No stage, a zero stride, widths that overflow, or a module the checkpoint lacks.
    pub fn load(config: &DacEncoderConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        if config.strides.is_empty() || config.strides.contains(&0) || config.channels == 0 {
            return Err(Error::shape(
                "DacEncoder",
                format!("{config:?} has no stage, a zero stride or no width"),
            ));
        }

        let mut width = config.channels;
        let input = Conv1d::load(&normalized(1, width, 7, 3, 1), vb.pp(0))?;

        let mut stages = Vec::with_capacity(config.strides.len());
        for (index, &stride) in config.strides.iter().enumerate() {
            let inner = width;
            width = width.checked_mul(2).ok_or_else(|| {
                Error::shape("DacEncoder", "the stage widths overflow".to_owned())
            })?;
            let stage = vb.pp(index + 1).pp("block");

            let units = DILATIONS
                .iter()
                .enumerate()
                .map(|(unit, &dilation)| {
                    let module = stage.pp(unit).pp("block");

                    Ok(Unit {
                        alphas: [alpha(module.pp(0), inner)?, alpha(module.pp(2), inner)?],
                        convs: [
                            Conv1d::load(
                                &normalized(inner, inner, 7, 3 * dilation, dilation),
                                module.pp(1),
                            )?,
                            Conv1d::load(&normalized(inner, inner, 1, 0, 1), module.pp(3))?,
                        ],
                    })
                })
                .collect::<Result<Vec<_>>>()?;

            stages.push(Stage {
                stride,
                units,
                alpha: alpha(stage.pp(3), inner)?,
                widen: Conv1d::load(
                    &ConvConfig {
                        stride: [stride],
                        ..normalized(inner, width, 2 * stride, stride.div_ceil(2), 1)
                    },
                    stage.pp(4),
                )?,
            });
        }

        let last = config.strides.len();
        Ok(Self {
            input,
            stages,
            alpha: alpha(vb.pp(last + 1), width)?,
            output: Conv1d::load(
                &normalized(width, config.latent_channels, 3, 1, 1),
                vb.pp(last + 2),
            )?,
        })
    }

    /// The samples one output frame covers: the product of the strides.
    pub fn hop(&self) -> usize {
        self.stages
            .iter()
            .map(|stage| stage.stride as usize)
            .product()
    }

    /// `samples`, `[batch, 1, length]`, encoded: `[batch, latent_channels, length / hop]` in
    /// their element type. `length` is a whole number of hops.
    ///
    /// # Errors
    ///
    /// A length that is not a whole number of hops, samples of another shape, or an operator's
    /// refusal.
    pub fn forward(&self, samples: &Tensor) -> Result<Tensor> {
        let hop = self.hop();
        match samples.dims() {
            &[_, 1, length] if length.is_multiple_of(hop) => {}
            dims => {
                return Err(Error::shape(
                    "DacEncoder",
                    format!("{dims:?}: a [batch, 1, samples] track of whole {hop}-sample hops"),
                ));
            }
        }

        let mut hidden = self.input.forward(samples)?;
        for stage in &self.stages {
            // A stage's units add into the residual stream.
            for unit in &stage.units {
                let first = unit.convs[0].forward(&functional::snake(&hidden, &unit.alphas[0])?)?;
                let second = unit.convs[1].forward(&functional::snake(&first, &unit.alphas[1])?)?;

                hidden = (hidden + second)?;
            }
            hidden = stage
                .widen
                .forward(&functional::snake(&hidden, &stage.alpha)?)?;
        }
        self.output
            .forward(&functional::snake(&hidden, &self.alpha)?)
    }
}

impl Module for DacEncoder {
    fn forward(&self, samples: &Tensor) -> tensor::Result<Tensor> {
        Ok(Self::forward(self, samples)?)
    }
}

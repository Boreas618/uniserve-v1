//! The modules of a causal video VAE's convolution stack: the per-frame normalization that
//! writes a convolution's padded input, and the residual block built from it.

use tensor::{Module, Tensor};

use crate::error::{Error, Result};
use crate::functional::{self, Padding};
use crate::modules::conv::{Conv3d, ConvConfig};
use crate::var_builder::VarBuilder;

/// What a checkpoint states of a [`FrameNorm`], and the causal pad of the frame it writes.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct FrameNormConfig {
    /// The channels it normalizes: the length of its weight and bias.
    pub channels: u32,
    /// The groups each frame's channels are normalized in.
    pub groups: u32,
    /// The variance floor.
    pub eps: f32,
    /// Whether the module states a `weight` and a `bias`.
    pub affine: bool,
    /// The rows and columns the written frame gains: top, bottom, left, right.
    pub spatial: [usize; 4],
    /// The zero frames the written clip gains before its first.
    pub front: usize,
    /// How the rows and columns beyond a frame's edge are filled.
    pub padding: Padding,
}

/// The group normalization of each frame of a clip, with the SiLU and the causal pad of the
/// convolution that reads it: `pad(silu(norm(x) · weight + bias))`
/// ([`functional::frame_norm_pad`]).
///
/// It differs from torch's `GroupNorm` in what it reduces and in what it writes: a group's
/// statistics are one frame's, not the clip's, and the output is activated and padded.
pub struct FrameNorm {
    /// The learned `(weight, bias)`, when the module states them.
    affine: Option<(Tensor, Tensor)>,
    config: FrameNormConfig,
}

impl FrameNorm {
    /// The normalization module at `vb`: its `weight` and `bias` of `config.channels` elements
    /// when the config states them.
    ///
    /// # Errors
    ///
    /// No groups, or a tensor the checkpoint lacks or holds at another width.
    pub fn load(config: &FrameNormConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        if config.groups == 0 {
            return Err(Error::shape(
                "FrameNorm",
                "a normalization of no groups has no statistics".to_owned(),
            ));
        }

        let channels = config.channels as usize;
        let affine = if config.affine {
            Some((vb.get(channels, "weight")?, vb.get(channels, "bias")?))
        } else {
            None
        };

        Ok(Self {
            affine,
            config: *config,
        })
    }

    /// `x`, `[n, channels, d, h, w]` in FP32, normalized, activated and padded:
    /// `[n, channels, d + front, h + top + bottom, w + left + right]`.
    ///
    /// # Errors
    ///
    /// A clip of another rank, element type or channel count, or the operators' refusals.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let config = &self.config;

        functional::frame_norm_pad(
            x,
            self.affine.as_ref().map(|(weight, bias)| (weight, bias)),
            config.groups,
            config.eps,
            config.spatial,
            config.front,
            config.padding,
        )
    }
}

impl Module for FrameNorm {
    fn forward(&self, x: &Tensor) -> tensor::Result<Tensor> {
        Ok(Self::forward(self, x)?)
    }
}

/// What a checkpoint states of a [`ResnetBlock3d`].
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ResnetBlock3dConfig {
    /// Channels the block reads.
    pub in_channels: u32,
    /// Channels the block writes.
    pub out_channels: u32,
    /// Group-normalization groups.
    pub groups: u32,
    /// Group-normalization variance floor.
    pub eps: f32,
    /// The rows and columns each normalization pads its frame by (top, bottom, left, right);
    /// a convolution of `top + bottom + 1` taps keeps the extent.
    pub spatial: [usize; 4],
    /// The zero frames each normalization writes before its clip.
    pub front: usize,
    /// How the rows and columns beyond a frame's edge are filled.
    pub padding: Padding,
    /// The convolutions' taps along each axis.
    pub kernel: u32,
    /// Whether cuDNN may use TF32 for the block's FP32 convolutions.
    pub tf32: bool,
}

/// A causal video residual block (diffusers' `ResnetBlock3D` with causal convolutions): two
/// stages of per-frame group normalization, SiLU, causal pad and convolution, and the input, or
/// its one-by-one projection where the width changes, added to the second stage's output.
///
/// ```text
/// h   = conv1(pad(silu(norm1(x))))
/// out = conv2(pad(silu(norm2(h)))) + shortcut(x)
/// ```
///
/// Each normalization and its pad are one [`FrameNorm`], each convolution carries its bias, and
/// the sum rounds once.
pub struct ResnetBlock3d {
    norm1: FrameNorm,
    conv1: Conv3d,
    norm2: FrameNorm,
    conv2: Conv3d,
    /// `conv_shortcut`, the projection of the input when the width changes.
    shortcut: Option<Conv3d>,
}

impl ResnetBlock3d {
    /// The block at `vb`: `norm1`, `conv1`, `norm2`, `conv2` and, when the width changes, the
    /// one-by-one `conv_shortcut`, each with its weight and bias.
    ///
    /// # Errors
    ///
    /// A config the modules refuse, or a tensor the checkpoint lacks or holds at another extent.
    pub fn load(config: &ResnetBlock3dConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        let norm = |channels: u32| FrameNormConfig {
            channels,
            groups: config.groups,
            eps: config.eps,
            affine: true,
            spatial: config.spatial,
            front: config.front,
            padding: config.padding,
        };
        let conv = |from: u32, to: u32, kernel: u32| ConvConfig {
            tf32: config.tf32,
            ..ConvConfig::new(from, to, [kernel; 3])
        };
        let (from, to) = (config.in_channels, config.out_channels);

        let shortcut = if from == to {
            None
        } else {
            Some(Conv3d::load(&conv(from, to, 1), vb.pp("conv_shortcut"))?)
        };

        Ok(Self {
            norm1: FrameNorm::load(&norm(from), vb.pp("norm1"))?,
            conv1: Conv3d::load(&conv(from, to, config.kernel), vb.pp("conv1"))?,
            norm2: FrameNorm::load(&norm(to), vb.pp("norm2"))?,
            conv2: Conv3d::load(&conv(to, to, config.kernel), vb.pp("conv2"))?,
            shortcut,
        })
    }

    /// The block's output for `x` `[n, in_channels, d, h, w]` in FP32:
    /// `[n, out_channels, d, h, w]`.
    ///
    /// # Errors
    ///
    /// A clip the normalizations or the convolutions refuse.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = self.conv1.forward(&self.norm1.forward(x)?)?;
        let update = self.conv2.forward(&self.norm2.forward(&hidden)?)?;

        // `(conv2 + bias) + residual`: the reference adds the convolution's bias in a pass of
        // its own and then the residual, and IEEE addition commutes, so this is the same
        // rounded sum.
        let residual = match &self.shortcut {
            Some(shortcut) => shortcut.forward(x)?,
            None => x.clone(),
        };
        Ok((update + residual)?)
    }
}

impl Module for ResnetBlock3d {
    fn forward(&self, x: &Tensor) -> tensor::Result<Tensor> {
        Ok(Self::forward(self, x)?)
    }
}

//! Convolution modules with torch's arguments: `Conv1d`, `ConvTranspose1d` and `Conv3d`, each a
//! filter and an optional bias over the functions of [`crate::functional`].
//!
//! A module's [`ConvConfig`] states what its torch module states: channels, kernel, stride, zero
//! padding, dilation, groups and whether it has a bias. Two fields are the checkpoint's and the
//! model's own: `weight_norm`, for a filter stored as torch's weight-norm pair `weight_v` and
//! `weight_g`, folded once at load; and `tf32`, the model's numerical declaration for an FP32
//! convolution.

use uniserve_tensor::{Module, Tensor};

use crate::error::{Error, Result};
use crate::functional;
use crate::var_builder::VarBuilder;

/// A convolution's torch arguments over `N` spatial axes.
///
/// [`ConvConfig::new`] states torch's defaults (stride one, no padding, no dilation, one group,
/// a bias), and a caller overrides the fields its module states with struct update syntax.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ConvConfig<const N: usize> {
    /// Input channels.
    pub in_channels: u32,
    /// Output channels.
    pub out_channels: u32,
    /// The filter's extent along each spatial axis.
    pub kernel: [u32; N],
    /// The step between output positions along each spatial axis.
    pub stride: [u32; N],
    /// The zero padding before (and after) the input along each spatial axis.
    pub padding: [u32; N],
    /// The spacing between filter taps along each spatial axis.
    pub dilation: [u32; N],
    /// How many channel groups the filter is split into.
    pub groups: u32,
    /// Whether the module states a `bias`.
    pub bias: bool,
    /// Whether the checkpoint stores the weight as torch's weight-norm pair, folded at load.
    pub weight_norm: bool,
    /// Whether cuDNN may use TF32 for an FP32 convolution.
    pub tf32: bool,
}

impl<const N: usize> ConvConfig<N> {
    /// `in_channels` to `out_channels` with a `kernel` filter and torch's other defaults.
    pub const fn new(in_channels: u32, out_channels: u32, kernel: [u32; N]) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel,
            stride: [1; N],
            padding: [0; N],
            dilation: [1; N],
            groups: 1,
            bias: true,
            weight_norm: false,
            tf32: false,
        }
    }

    /// The filter's extents: torch's `[out, in / groups, kernel…]`, or `[in, out / groups,
    /// kernel…]` for a transposed convolution.
    ///
    /// # Errors
    ///
    /// When the config is no convolution: no groups, no channels, or channels the groups do not
    /// divide.
    fn filter(&self, owner: &'static str, transposed: bool) -> Result<Vec<usize>> {
        if self.groups == 0
            || self.in_channels == 0
            || self.out_channels == 0
            || !self.in_channels.is_multiple_of(self.groups)
            || !self.out_channels.is_multiple_of(self.groups)
        {
            return Err(Error::shape(
                owner,
                format!("{self:?} is not a convolution"),
            ));
        }

        let (first, second) = match transposed {
            true => (self.in_channels, self.out_channels / self.groups),
            false => (self.out_channels, self.in_channels / self.groups),
        };
        let mut extents = vec![first as usize, second as usize];
        extents.extend(self.kernel.iter().map(|taps| *taps as usize));
        Ok(extents)
    }

    /// The module's filter and bias at `vb`: `weight` (or its weight-norm pair) and, when the
    /// config states one, `bias`.
    fn load(
        &self,
        owner: &'static str,
        vb: &VarBuilder<'_, '_>,
        transposed: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let filter = self.filter(owner, transposed)?;

        let weight = if self.weight_norm {
            vb.weight_norm("weight")?
        } else {
            vb.get(filter.as_slice(), "weight")?
        };
        let bias = if self.bias {
            Some(vb.get(self.out_channels as usize, "bias")?)
        } else {
            None
        };

        self.check(owner, &weight, bias.as_ref(), transposed)?;
        Ok((weight, bias))
    }

    /// Checks that `weight` and `bias` are the filter and the bias this config states.
    fn check(
        &self,
        owner: &'static str,
        weight: &Tensor,
        bias: Option<&Tensor>,
        transposed: bool,
    ) -> Result<()> {
        let filter = self.filter(owner, transposed)?;
        if weight.dims() != filter {
            return Err(Error::shape(
                owner,
                format!(
                    "a filter of {:?} for a config whose filter is {filter:?}",
                    weight.dims()
                ),
            ));
        }

        if self.bias != bias.is_some()
            || bias.is_some_and(|bias| bias.dims() != [self.out_channels as usize])
        {
            return Err(Error::shape(
                owner,
                format!(
                    "the bias does not match a config of {} output channels with bias {}",
                    self.out_channels, self.bias
                ),
            ));
        }
        Ok(())
    }
}

/// A one-dimensional convolution: torch's `Conv1d` ([`functional::conv1d`]).
pub struct Conv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    config: ConvConfig<1>,
}

impl Conv1d {
    /// The module at `vb`: `weight` `[out, in / groups, k]` (or its weight-norm pair) and, when
    /// the config states one, `bias` `[out]`.
    ///
    /// # Errors
    ///
    /// A config no convolution has, or a tensor the checkpoint lacks or holds at another extent.
    pub fn load(config: &ConvConfig<1>, vb: VarBuilder<'_, '_>) -> Result<Self> {
        let (weight, bias) = config.load("Conv1d", &vb, false)?;

        Ok(Self {
            weight,
            bias,
            config: *config,
        })
    }

    /// The convolution over weights a model computes instead of reading.
    ///
    /// # Errors
    ///
    /// As [`Self::load`], for the weights given.
    pub fn new(weight: Tensor, bias: Option<Tensor>, config: &ConvConfig<1>) -> Result<Self> {
        config.check("Conv1d", &weight, bias.as_ref(), false)?;

        Ok(Self {
            weight,
            bias,
            config: *config,
        })
    }

    /// `conv1d(x)` on `[n, in, l]` sequences: `[n, out, l']` in `x`'s element type.
    ///
    /// # Errors
    ///
    /// A sequence of other channels, one too short to leave an output, or the operator's
    /// refusal.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let config = &self.config;

        functional::conv1d(
            x,
            &self.weight,
            self.bias.as_ref(),
            config.stride[0],
            config.padding[0],
            config.dilation[0],
            config.groups,
            config.tf32,
        )
    }
}

impl Module for Conv1d {
    fn forward(&self, x: &Tensor) -> uniserve_tensor::Result<Tensor> {
        Ok(Self::forward(self, x)?)
    }
}

/// A one-dimensional transposed convolution: torch's `ConvTranspose1d` without output padding
/// ([`functional::conv_transpose1d`]).
pub struct ConvTranspose1d {
    weight: Tensor,
    bias: Option<Tensor>,
    config: ConvConfig<1>,
}

impl ConvTranspose1d {
    /// The module at `vb`: `weight` `[in, out / groups, k]` (or its weight-norm pair) and, when
    /// the config states one, `bias` `[out]`.
    ///
    /// # Errors
    ///
    /// A config no convolution has, or a tensor the checkpoint lacks or holds at another extent.
    pub fn load(config: &ConvConfig<1>, vb: VarBuilder<'_, '_>) -> Result<Self> {
        let (weight, bias) = config.load("ConvTranspose1d", &vb, true)?;

        Ok(Self {
            weight,
            bias,
            config: *config,
        })
    }

    /// `conv_transpose1d(x)` on `[n, in, l]` sequences: `[n, out, l']` in `x`'s element type.
    ///
    /// # Errors
    ///
    /// A sequence of other channels, or the operator's refusal.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let config = &self.config;

        functional::conv_transpose1d(
            x,
            &self.weight,
            self.bias.as_ref(),
            config.stride[0],
            config.padding[0],
            config.dilation[0],
            config.groups,
            config.tf32,
        )
    }
}

impl Module for ConvTranspose1d {
    fn forward(&self, x: &Tensor) -> uniserve_tensor::Result<Tensor> {
        Ok(Self::forward(self, x)?)
    }
}

/// A three-dimensional convolution: torch's `Conv3d` ([`functional::conv3d`]).
pub struct Conv3d {
    weight: Tensor,
    bias: Option<Tensor>,
    config: ConvConfig<3>,
}

impl Conv3d {
    /// The module at `vb`: `weight` `[out, in / groups, kd, kh, kw]` and, when the config states
    /// one, `bias` `[out]`.
    ///
    /// # Errors
    ///
    /// A config no convolution has, or a tensor the checkpoint lacks or holds at another extent.
    pub fn load(config: &ConvConfig<3>, vb: VarBuilder<'_, '_>) -> Result<Self> {
        let (weight, bias) = config.load("Conv3d", &vb, false)?;

        Ok(Self {
            weight,
            bias,
            config: *config,
        })
    }

    /// The convolution over weights a model computes instead of reading.
    ///
    /// # Errors
    ///
    /// As [`Self::load`], for the weights given.
    pub fn new(weight: Tensor, bias: Option<Tensor>, config: &ConvConfig<3>) -> Result<Self> {
        config.check("Conv3d", &weight, bias.as_ref(), false)?;

        Ok(Self {
            weight,
            bias,
            config: *config,
        })
    }

    /// `conv3d(x)`: `x` `[n, in, d, h, w]` to `[n, out, d', h', w']` in `x`'s element type and
    /// storage order: a frame stored channels-last is read and written channels-last
    /// ([`functional::conv3d`]).
    ///
    /// # Errors
    ///
    /// A frame of other channels, one too small to leave an output, or the operator's refusal.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let config = &self.config;

        functional::conv3d(
            x,
            &self.weight,
            self.bias.as_ref(),
            config.stride,
            config.padding,
            config.dilation,
            config.groups,
            config.tf32,
        )
    }
}

impl Module for Conv3d {
    fn forward(&self, x: &Tensor) -> uniserve_tensor::Result<Tensor> {
        Ok(Self::forward(self, x)?)
    }
}

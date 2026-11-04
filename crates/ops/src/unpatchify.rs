//! `unpatchify_video`: packed latent tokens into a planar raster, and back out again.
//!
//! The video VAE's decoder does not read the DiT's packed rows: it reads a `[channels, frames,
//! height, width]` raster, and the decoder's own output is the same kind of raster with a wider
//! patch. One operator covers both directions the model needs, because they are one index arithmetic
//! with different patch shapes (`uniserve_kernels/src/uniserve_kernels/patch.py::_unpatchify_video_tokens_kernel`,
//! ported as it is):
//!
//! ```text
//! tokens [(frames·height·width), channels · patch_volume]
//! out    [channels, frames · patch_frames, height · patch_height, width · patch_width]
//!
//! token (t, y, x), element c · pf·ph·pw + pt·ph·pw + ph_i·pw + pw_i
//!   -> out[c, t·pf + pt, y·ph + ph_i, x·pw + pw_i]
//!
//! with `affine`:  out = value · scale[c] + bias[c]
//! ```
//!
//! Every product and sum is FP32 and the result rounds once into `dtype`, which is why the tolerance
//! is a single rounding. Tokens past the grid are never read: the launch walks the *output*, so a
//! token buffer with spare rows is not an error but its spare rows are invisible.
//!
//! Scale and bias are the checkpoint's `latents_std` and `latents_mean`, uploaded once as constants;
//! the decoder's output carries `affine = false` because `proj_out`'s epilogue already adds the bias.
//! They are `Option` views rather than null pointers: the descriptor's flag says which launch form
//! this is, and a launch that disagrees with it in either direction is refused.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, Strides, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// One compilation of `unpatchify_video`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Element type of the tokens, of the scale and bias, and of the output.
    pub dtype: DType,
    /// Channels of the raster.
    pub channels: u32,
    /// Token grid: frames, rows and columns of tokens.
    pub frames: u32,
    /// Token grid rows.
    pub height: u32,
    /// Token grid columns.
    pub width: u32,
    /// Voxels one token carries along frames.
    pub patch_frames: u32,
    /// Rows one token carries.
    pub patch_height: u32,
    /// Columns one token carries.
    pub patch_width: u32,
    /// Whether the launch denormalizes with a per-channel scale and bias.
    pub affine: bool,
    /// Elements one thread block writes.
    pub block: u32,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl Descriptor {
    /// Voxels one token carries: `patch_frames · patch_height · patch_width`.
    pub fn patch_volume(&self) -> u64 {
        u64::from(self.patch_frames) * u64::from(self.patch_height) * u64::from(self.patch_width)
    }

    /// Values one token row holds: `channels · patch_volume`.
    pub fn token_width(&self) -> u64 {
        u64::from(self.channels) * self.patch_volume()
    }

    /// Tokens in the grid.
    pub fn tokens(&self) -> u64 {
        u64::from(self.frames) * u64::from(self.height) * u64::from(self.width)
    }
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Every field is a shape or a flag: no raw float enters a key, and the affine form is its own
        // variant because the launch carries two more operands.
        Key::new()
            .field("dtype", self.dtype)
            .field("channels", self.channels)
            .field("frames", self.frames)
            .field("height", self.height)
            .field("width", self.width)
            .field(
                "patch",
                format_args!(
                    "{}x{}x{}",
                    self.patch_frames, self.patch_height, self.patch_width
                ),
            )
            .field("affine", self.affine)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one unpatchify.
#[derive(Clone, Copy)]
pub struct Args {
    /// `[tokens, channels · patch_volume]` the packed token rows.
    pub tokens: TensorView,
    /// `[channels]` the per-channel scale, present exactly when the descriptor's `affine` is set.
    pub scale: Option<TensorView>,
    /// `[channels]` the per-channel bias, present exactly when the descriptor's `affine` is set.
    pub bias: Option<TensorView>,
    /// `[channels, frames · patch_frames, height · patch_height, width · patch_width]` the raster.
    pub out: TensorView,
}

/// The packed-token to planar-raster mapping, with an optional per-channel denormalization.
pub struct Unpatchify;

impl Op for Unpatchify {
    const NAME: &'static str = "unpatchify_video";

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let channels = u64::from(descriptor.channels);
        let volume = descriptor.patch_volume();
        let tokens = descriptor.tokens();
        if channels == 0
            || descriptor.frames == 0
            || descriptor.height == 0
            || descriptor.width == 0
            || volume == 0
        {
            return Err(Error::argument(
                Self::NAME,
                "a grid with no tokens, no channels or a patch with no voxels has no raster"
                    .to_owned(),
            ));
        }
        let out_shape = Shape::new(&[
            channels,
            u64::from(descriptor.frames) * u64::from(descriptor.patch_frames),
            u64::from(descriptor.height) * u64::from(descriptor.patch_height),
            u64::from(descriptor.width) * u64::from(descriptor.patch_width),
        ]);
        args.out.require_shape(Self::NAME, out_shape)?;
        args.out
            .require_row(Self::NAME, descriptor.dtype, out_shape.dims()[0])?;
        args.tokens
            .require_shape(Self::NAME, Shape::new(&[tokens, descriptor.token_width()]))?;
        if args.tokens.dtype != descriptor.dtype {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {} tokens, found {}",
                    descriptor.dtype, args.tokens.dtype
                ),
            ));
        }
        // The two denormalization operands arrive together or not at all, and the descriptor says
        // which form this launch is: one alone would write a raster nobody asked for.
        let affine = match (args.scale, args.bias) {
            (Some(scale), Some(bias)) => Some((scale, bias)),
            (None, None) => None,
            _ => {
                return Err(Error::argument(
                    Self::NAME,
                    "the scale and the bias are given together or both absent".to_owned(),
                ));
            }
        };
        if affine.is_some() != descriptor.affine {
            return Err(Error::argument(
                Self::NAME,
                if descriptor.affine {
                    "the module was compiled with a per-channel affine and the launch has none"
                } else {
                    "the module was compiled without a per-channel affine and the launch has one"
                }
                .to_owned(),
            ));
        }
        if let Some((scale, bias)) = affine {
            for (name, view) in [("scale", scale), ("bias", bias)] {
                view.require_shape(Self::NAME, Shape::new(&[channels]))?;
                if view.dtype != descriptor.dtype {
                    return Err(Error::argument(
                        Self::NAME,
                        format!("expected {} {name}, found {}", descriptor.dtype, view.dtype),
                    ));
                }
            }
        }
        // The program indexes the raster as a flat row-major run of voxels, so a strided destination
        // would have it write the wrong voxels rather than refuse.
        if args.out.strides != Strides::row_major(args.out.shape) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the raster is written row-major; found strides {}",
                    args.out.strides
                ),
            ));
        }

        let mut arguments = Arguments::new();
        arguments.view(args.tokens);
        if let Some((scale, bias)) = affine {
            arguments.view(scale).view(bias);
        }
        arguments
            .view(args.out)
            .u32(descriptor.channels)
            .u32(descriptor.frames)
            .u32(descriptor.height)
            .u32(descriptor.width)
            .u32(descriptor.patch_frames)
            .u32(descriptor.patch_height)
            .u32(descriptor.patch_width)
            .flag(descriptor.affine)
            .u32(descriptor.block)
            .flag(descriptor.pdl);
        // The grid walks the output's elements: one thread writes one voxel, so a token's voxels are
        // written by the threads that own them and no thread waits for another.
        kernel.launch(
            LaunchGrid::linear(args.out.numel().div_ceil(u64::from(descriptor.block)))?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        Args {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // Any number of voxels may share a destination only if the grid repeats a token, which the
        // index arithmetic does not; each written value is one FP32 product and sum, rounded once.
        let _ = descriptor;
        Tolerance::rounding(1, DType::F32, descriptor.dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn latents() -> Descriptor {
        Descriptor {
            dtype: DType::F16,
            channels: 24,
            frames: 37,
            height: 24,
            width: 42,
            patch_frames: 1,
            patch_height: 2,
            patch_width: 2,
            affine: true,
            block: 1024,
            pdl: false,
        }
    }

    #[test]
    fn the_patch_volume_and_the_token_width_follow_the_patch() {
        assert_eq!(latents().patch_volume(), 4);
        assert_eq!(latents().token_width(), 96);
        assert_eq!(latents().tokens(), 37 * 24 * 42);
        // The decoder's own output: three channels and a 4x16x16 patch.
        let decoded = Descriptor {
            channels: 3,
            frames: 9,
            height: 8,
            width: 8,
            patch_frames: 4,
            patch_height: 16,
            patch_width: 16,
            affine: false,
            ..latents()
        };
        assert_eq!(decoded.patch_volume(), 1024);
        assert_eq!(decoded.token_width(), 3 * 1024);
    }

    #[test]
    fn the_affine_form_is_its_own_key() {
        assert_eq!(
            latents().to_string(),
            "dtype_f16_channels_24_frames_37_height_24_width_42_patch_1x2x2_affine_true_block_1024_pdl_false"
        );
        assert_ne!(
            latents().to_string(),
            Descriptor {
                affine: false,
                ..latents()
            }
            .to_string()
        );
    }

    #[test]
    fn the_bound_is_the_one_rounding() {
        assert_eq!(
            Unpatchify::tolerance(&latents()),
            Tolerance::rounding(1, DType::F32, DType::F16)
        );
    }
}

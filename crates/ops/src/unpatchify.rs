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
//! The descriptor carries no geometry: the program reads the channels, the token grid and the patch
//! as launch words, so one compiled module serves every raster of an element type. Its key names the
//! element type, the affine form (whose launch carries two more pointers), the block and the PDL
//! option — the values that do select a program — so a canvas or a patch is never a reason to compile
//! a second module, and a shape the manifest did not guess cannot arrive as a missing kernel.
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
    /// Whether the launch denormalizes with a per-channel scale and bias.
    pub affine: bool,
    /// Elements one thread block writes.
    pub block: u32,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Every field is a shape or a flag: no raw float enters a key, and the affine form is its own
        // variant because the launch carries two more operands.
        Key::new()
            .field("dtype", self.dtype)
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
    /// The token grid: frames, rows and columns. A request's grid varies — a clip's length and its
    /// canvas are the request's — so it is a launch value and not a compiled one.
    pub frames: u32,
    /// Token grid rows.
    pub height: u32,
    /// Token grid columns.
    pub width: u32,
    /// Channels of the raster.
    pub channels: u32,
    /// Voxels one token carries along frames.
    pub patch_frames: u32,
    /// Rows one token carries.
    pub patch_height: u32,
    /// Columns one token carries.
    pub patch_width: u32,
}

impl Args {
    /// Voxels one token carries: `patch_frames · patch_height · patch_width`.
    pub fn patch_volume(&self) -> u64 {
        u64::from(self.patch_frames) * u64::from(self.patch_height) * u64::from(self.patch_width)
    }

    /// Values one token row holds: `channels · patch_volume`.
    pub fn token_width(&self) -> u64 {
        u64::from(self.channels) * self.patch_volume()
    }

    /// Tokens in a grid of `frames × height × width` token rows.
    pub fn tokens(&self) -> u64 {
        u64::from(self.frames) * u64::from(self.height) * u64::from(self.width)
    }

    /// The planar raster a launch writes: `[channels, frames, height, width]`, each spatial axis
    /// multiplied by the patch it replaces.
    ///
    /// The channels are the **outer** axis and the width the **innermost**, so the contiguous row a
    /// launch may write is `width · patch_width` values — not `channels`. The two differ in every
    /// geometry this operator serves (`24` against `84` for the transformer's latents, `3` against
    /// `160` for the decoder's raster), and a launch check that reads the outer axis for the row
    /// refuses each of them.
    pub fn raster(&self) -> Shape {
        Shape::new(&[
            u64::from(self.channels),
            u64::from(self.frames) * u64::from(self.patch_frames),
            u64::from(self.height) * u64::from(self.patch_height),
            u64::from(self.width) * u64::from(self.patch_width),
        ])
    }

    /// Values on the raster's innermost axis, which is the row a launch writes.
    pub fn raster_row(&self) -> u64 {
        u64::from(self.width) * u64::from(self.patch_width)
    }

    /// Whether this launch may run at all: the operands' shapes, their element types, the affine
    /// pair's presence against the compiled form, and the destination's contiguity.
    ///
    /// It is separate from `launch` so that the contract is reachable without a device: a kernel is
    /// only loadable on one, and a check that cannot be run is a check nobody has run.
    pub fn validate(&self, descriptor: &Descriptor) -> Result<()> {
        let channels = u64::from(self.channels);
        let volume = self.patch_volume();
        let frames = u64::from(self.frames);
        let height = u64::from(self.height);
        let width = u64::from(self.width);
        let tokens = self.tokens();
        if channels == 0 || frames == 0 || height == 0 || width == 0 || volume == 0 {
            return Err(Error::argument(
                Unpatchify::NAME,
                "a grid with no tokens, no channels or a patch with no voxels has no raster"
                    .to_owned(),
            ));
        }
        let out_shape = self.raster();
        self.out.require_shape(Unpatchify::NAME, out_shape)?;
        self.out
            .require_row(Unpatchify::NAME, descriptor.dtype, self.raster_row())?;
        self.tokens
            .require_shape(Unpatchify::NAME, Shape::new(&[tokens, self.token_width()]))?;
        if self.tokens.dtype != descriptor.dtype {
            return Err(Error::argument(
                Unpatchify::NAME,
                format!(
                    "expected {} tokens, found {}",
                    descriptor.dtype, self.tokens.dtype
                ),
            ));
        }
        // The two denormalization operands arrive together or not at all, and the descriptor says
        // which form this launch is: one alone would write a raster nobody asked for.
        let affine = match (self.scale, self.bias) {
            (Some(scale), Some(bias)) => Some((scale, bias)),
            (None, None) => None,
            _ => {
                return Err(Error::argument(
                    Unpatchify::NAME,
                    "the scale and the bias are given together or both absent".to_owned(),
                ));
            }
        };
        if affine.is_some() != descriptor.affine {
            return Err(Error::argument(
                Unpatchify::NAME,
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
                view.require_shape(Unpatchify::NAME, Shape::new(&[channels]))?;
                if view.dtype != descriptor.dtype {
                    return Err(Error::argument(
                        Unpatchify::NAME,
                        format!("expected {} {name}, found {}", descriptor.dtype, view.dtype),
                    ));
                }
            }
        }
        // The raster is written contiguously: the program addresses the destination as a flat
        // element range, so a destination whose rows interleave would be written to the wrong voxels.
        // A tile inside a larger raster needs the three outer strides as launch words, which the
        // program does not take yet; refusing here is what keeps that from being a silent wrong answer.
        if self.out.strides != Strides::row_major(out_shape) {
            return Err(Error::argument(
                Unpatchify::NAME,
                format!(
                    "the raster is written contiguously; found strides {} for {out_shape}",
                    self.out.strides
                ),
            ));
        }
        Ok(())
    }
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
        args.validate(descriptor)?;
        let mut arguments = Arguments::new();
        arguments.view(args.tokens);
        if let (Some(scale), Some(bias)) = (args.scale, args.bias) {
            arguments.view(scale).view(bias);
        }
        // The affine operand pair and the raster's form are the compilation's, not the launch's: the
        // descriptor says which program this is, so naming it here would only shift the two scratch
        // addresses the compiler's launcher appends after these parameters.
        arguments
            .view(args.out)
            .u32(args.channels)
            .u32(args.frames)
            .u32(args.height)
            .u32(args.width)
            .u32(args.patch_frames)
            .u32(args.patch_height)
            .u32(args.patch_width)
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
            affine: true,
            block: 1024,
            pdl: false,
        }
    }

    /// A contiguous view of `dims`, which is what every operand of an admitted launch is.
    fn contiguous(dtype: DType, dims: &[u64]) -> TensorView {
        let shape = Shape::new(dims);
        TensorView::with_strides(
            crate::tensor::DevicePtr::new(0x2000),
            dtype,
            shape,
            Strides::row_major(shape),
        )
    }

    fn grid() -> Args {
        let view = TensorView::with_strides(
            crate::tensor::DevicePtr::new(0x1000),
            DType::F16,
            Shape::new(&[1, 1]),
            Strides::new(&[1, 1]),
        );
        Args {
            tokens: view,
            scale: None,
            bias: None,
            out: view,
            frames: 37,
            height: 24,
            width: 42,
            channels: 24,
            patch_frames: 1,
            patch_height: 2,
            patch_width: 2,
        }
    }

    #[test]
    fn the_patch_volume_and_the_token_width_follow_the_patch() {
        assert_eq!(grid().patch_volume(), 4);
        assert_eq!(grid().token_width(), 96);
        assert_eq!(grid().tokens(), 37 * 24 * 42);
        // The decoder's own output: three channels and a 4x16x16 patch, in the same compiled module.
        let decoded = Args {
            channels: 3,
            patch_frames: 4,
            patch_height: 16,
            patch_width: 16,
            frames: 9,
            height: 8,
            width: 10,
            ..grid()
        };
        assert_eq!(decoded.patch_volume(), 1024);
        assert_eq!(decoded.token_width(), 3 * 1024);
        assert_eq!(decoded.tokens(), 9 * 8 * 10);
        // Neither the grid nor the patch reaches the key: the program reads both as words, so the
        // decoder's raster and the transformer's are one compiled module.
        let key = latents().to_string();
        assert!(!key.contains("patch"), "{key}");
        assert!(!key.contains("channels"), "{key}");
    }

    #[test]
    fn the_raster_row_is_the_width_and_not_the_channels() {
        // The transformer's latents: 24 channels out of a `1x2x2` patch, so the raster is
        // `[24, 37, 48, 84]` and its innermost axis is 84 values — the row a launch writes.
        let latents = grid();
        assert_eq!(latents.raster().dims(), &[24, 37, 48, 84]);
        assert_eq!(latents.raster_row(), 84);
        assert_ne!(latents.raster_row(), u64::from(latents.channels));

        // The decoder's raster from the same compiled module: 3 channels and a `4x16x16` patch.
        let decoded = Args {
            channels: 3,
            patch_frames: 4,
            patch_height: 16,
            patch_width: 16,
            frames: 9,
            height: 8,
            width: 10,
            ..latents
        };
        assert_eq!(decoded.raster().dims(), &[3, 36, 128, 160]);
        assert_eq!(decoded.raster_row(), 160);

        // Neither geometry has a raster row equal to its channel count, which is why a launch check
        // reading the outer axis refuses every real raster instead of admitting it.
        assert_ne!(decoded.raster_row(), u64::from(decoded.channels));
    }

    #[test]
    fn the_launch_refuses_a_raster_whose_row_is_its_channel_count() {
        let descriptor = latents();
        let mut launch = grid();
        launch.tokens = contiguous(DType::F16, &[launch.tokens(), launch.token_width()]);
        let per_channel = contiguous(DType::F16, &[u64::from(launch.channels)]);
        launch.scale = Some(per_channel);
        launch.bias = Some(per_channel);

        // The raster this operator writes: `[channels, frames, height, width]` and its innermost
        // axis is the width, so the launch is admitted.
        launch.out = contiguous(DType::F16, &[24, 37, 48, 84]);
        launch
            .validate(&descriptor)
            .expect("the raster the operator writes is admitted");

        // The raster rowed by its channel count, which is what a check reading the outer axis
        // demands, is refused — and the refusal names the raster the operator writes, not the one
        // the check would have accepted. `require_shape` holds that shape before the row check, so
        // the message a caller sees is the raster's own dimensions.
        launch.out = contiguous(DType::F16, &[24, 37, 48, 24]);
        let error = launch
            .validate(&descriptor)
            .expect_err("a raster whose row is the channel count");
        let text = error.to_string();
        assert!(text.contains("[24, 37, 48, 84]"), "{text}");

        // The row check itself: the raster's shape is right and its innermost axis is not contiguous,
        // so the program would address the wrong voxels. This is the branch `require_row` adds over
        // `require_shape`, and it is the one a tile inside a larger raster would hit.
        let shape = Shape::new(&[24, 37, 48, 84]);
        launch.out = TensorView::with_strides(
            crate::tensor::DevicePtr::new(0x3000),
            DType::F16,
            shape,
            Strides::new(&[1, 24, 24 * 37, 24 * 37 * 48]),
        );
        let error = launch
            .validate(&descriptor)
            .expect_err("a raster written column-major");
        let text = error.to_string();
        assert!(text.contains("last axis"), "{text}");
    }

    #[test]
    fn the_affine_form_is_its_own_key() {
        assert_eq!(
            latents().to_string(),
            "dtype_f16_affine_true_block_1024_pdl_false"
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

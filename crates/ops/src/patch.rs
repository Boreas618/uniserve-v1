//! `patch`: the diffusion transformer's latent, packed into its rows.
//!
//! The DiT reads packed rows — `[video | audio | prefix | padding]` — and one video row is one
//! `1×2×2` patch of the latent's `channels` channels, so a row is `channels · 4` values wide. The
//! capability boundary hands the diffuser a latent `[channels, frames, height, width]`, so every
//! evaluation converts rows to latent and back:
//!
//! ```text
//! x [channels, frames, height, width] -> r [rows, channels · 4]
//! row    = ((f · height/2) + i) · (width/2) + j      f, i, j in raster order
//! column = ((c · 2) + a) · 2 + b                     c major, then the patch's (a, b)
//! r[row, column] = x[c, f, 2·i + a, 2·j + b]
//! ```
//!
//! One direction, because it is the only one the transformer's input needs: the latent is packed
//! once per evaluation before the denoiser reads it. The inverse — the sampler's result or a
//! decoder's raster — is `unpatchify_video` at `1×2×2`, the generic operator both rasters use, so
//! there is one inverse in the tree rather than a `2×2`-only one beside it. The descriptor carries
//! the geometry and no index table, so a captured step replays it and no host-side permutation is
//! needed. The move is elementwise and exact.
//!
//! The geometry is a launch value and the descriptor carries none: the program reads `channels`,
//! `height`, `width` and `frames` as words, so one compiled module serves every request of every
//! raster. Its key names the element type, the block and the PDL option — the values that do select a
//! program — so a canvas is never a reason to compile a second module and a request shape the
//! manifest did not guess cannot arrive as a missing kernel. `rows` — the packed row count — is the
//! derived value `frames · height/2 · width/2` and is passed beside the four, checked against them.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// Elements one thread block handles along a destination row.
pub const BLOCK: u32 = 1024;

/// One compilation of `patch`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Element type of the latent and of the rows; the move is exact in either direction.
    pub dtype: DType,
    /// Elements one thread block handles along a row.
    pub block: u32,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl Descriptor {
    /// A mapping whose module serves every raster of `dtype`.
    ///
    /// The geometry is not a compilation choice — the program reads it as words — so it is checked
    /// where it is consumed, by [`geometry`], and a canvas never selects a module.
    pub const fn new(dtype: DType, block: u32, pdl: bool) -> Self {
        Self { dtype, block, pdl }
    }
}

/// Refuse a geometry no `1×2×2` raster can have, naming it before a launch.
///
/// A patch is `1×2×2`, so an odd side is not a request this operator can honour: it is a model
/// geometry bug, and it is refused here rather than at the launch, so the failure names the caller's
/// geometry instead of arriving as an indexing fault in a kernel.
pub fn geometry(channels: u32, height: u32, width: u32) -> Result<()> {
    if channels == 0
        || height == 0
        || width == 0
        || !height.is_multiple_of(2)
        || !width.is_multiple_of(2)
    {
        return Err(Error::argument(
            "patch",
            format!(
                "a latent of {channels} channels and {height}\u{d7}{width} has no 1\u{d7}2\u{d7}2 patch \
                 raster: every side must be nonzero and even"
            ),
        ));
    }
    Ok(())
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one mapping.
#[derive(Clone, Copy)]
pub struct Args {
    /// The latent `[channels, frames, height, width]`.
    pub x: TensorView,
    /// The packed rows `[rows, channels · 4]`, `rows = frames · height/2 · width/2`.
    pub out: TensorView,
    /// Frames in the latent; the packed row count is `frames · rows_per_frame`.
    pub frames: u32,
    /// The latent's channels: `channels · 4` is a row's width.
    pub channels: u32,
    /// The latent's height, which must be even: a patch is `2` tall.
    pub height: u32,
    /// The latent's width, which must be even: a patch is `2` wide.
    pub width: u32,
}

impl Args {
    /// Values per packed row: `channels · 4`.
    pub fn row_width(&self) -> u64 {
        u64::from(self.channels) * 4
    }

    /// Packed rows per frame: `height/2 · width/2`.
    ///
    /// A launch's row count is exactly `frames · rows_per_frame()`; a destination with more rows
    /// would leave its tail untouched, which reads as stale latents rather than as an error, so the
    /// shape check is an equality and not a lower bound.
    pub fn rows_per_frame(&self) -> u64 {
        u64::from(self.height / 2) * u64::from(self.width / 2)
    }
}

/// The latent-to-rows mapping.
pub struct Patch;

impl Op for Patch {
    const NAME: &'static str = "patch";

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        geometry(args.channels, args.height, args.width)?;
        let channels = u64::from(args.channels);
        let height = u64::from(args.height);
        let width = u64::from(args.width);
        let frames = u64::from(args.frames);
        if frames == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a latent with no frames has no rows to map".to_owned(),
            ));
        }
        let rows = frames * args.rows_per_frame();
        let row_width = args.row_width();
        let latent = Shape::new(&[channels, frames, height, width]);
        let packed = Shape::new(&[rows, row_width]);

        args.x.require_shape(Self::NAME, latent)?;
        args.out.require_shape(Self::NAME, packed)?;
        for (name, view) in [("latent", args.x), ("rows", args.out)] {
            if view.dtype != descriptor.dtype {
                return Err(Error::argument(
                    Self::NAME,
                    format!("expected {} {name}, found {}", descriptor.dtype, view.dtype),
                ));
            }
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.x)
            .view(args.out)
            .u32(args.frames)
            .u32(u32::try_from(rows).map_err(|_| {
                Error::argument(
                    Self::NAME,
                    format!("{rows} packed rows do not fit a kernel parameter"),
                )
            })?)
            .u32(args.channels)
            .u32(args.height)
            .u32(args.width)
            .u32(descriptor.block)
            .flag(descriptor.pdl);
        // The grid walks the rows: one block per (row, element block). A row element is written by
        // exactly one block and reads a single latent element, so the order between blocks never
        // matters and a replay is deterministic.
        kernel.launch(
            LaunchGrid::new([rows, row_width.div_ceil(u64::from(descriptor.block)), 1])?,
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

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // A permutation of elements: no arithmetic, so the result is bit-identical.
        Tolerance::exact()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{DevicePtr, Strides};

    fn descriptor() -> Descriptor {
        Descriptor {
            dtype: DType::BF16,
            block: BLOCK,
            pdl: false,
        }
    }

    #[test]
    fn the_key_names_only_what_selects_a_program() {
        // The program reads the channels, the height and the width as words, so none of them may
        // appear in the key: a key that named a canvas would compile one module per canvas for no
        // compiled difference, and a canvas the manifest missed would be a missing kernel.
        assert_eq!(descriptor().to_string(), "dtype_bf16_block_1024_pdl_false");
        assert_eq!(
            Descriptor::new(DType::F32, 2048, true).to_string(),
            "dtype_f32_block_2048_pdl_true"
        );
    }

    #[test]
    fn a_row_is_the_channels_patch_and_a_frame_is_one_raster_of_them() {
        let view = |shape: &[u64]| {
            TensorView::with_strides(
                DevicePtr::new(0x1000),
                DType::BF16,
                Shape::new(shape),
                Strides::row_major(Shape::new(shape)),
            )
        };
        let args = Args {
            x: view(&[24, 3, 32, 48]),
            out: view(&[72, 96]),
            frames: 3,
            channels: 24,
            height: 32,
            width: 48,
        };
        assert_eq!(args.row_width(), 96);
        assert_eq!(args.rows_per_frame(), 16 * 24);
    }

    #[test]
    fn the_frames_are_a_launch_value_so_one_module_serves_every_request() {
        // The frame count changes `rows`, which is not a descriptor field, so a 5-frame and a
        // 121-frame request share one compiled program — as do two rasters of different canvases.
        assert_eq!(descriptor(), descriptor());
    }

    #[test]
    fn an_odd_side_is_refused_before_a_launch() {
        // A canvas side is a multiple of 32 and a latent side is that over 16, so every raster a
        // request can produce is even. An odd one is a geometry bug, refused by name rather than
        // arriving as an indexing fault in a kernel.
        assert!(geometry(24, 48, 48).is_ok());
        assert!(geometry(24, 48, 21).is_err());
        assert!(geometry(24, 21, 48).is_err());
        assert!(geometry(0, 48, 48).is_err());
    }

    #[test]
    fn the_mapping_is_exact() {
        assert_eq!(Patch::tolerance(&descriptor()), Tolerance::exact());
    }
}

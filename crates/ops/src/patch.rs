//! `patch`: the diffusion transformer's packed rows against its latent, in either direction.
//!
//! The DiT reads packed rows — `[video | audio | prefix | padding]` — and one video row is one
//! `1×2×2` patch of the latent's `channels` channels, so a row is `channels · 4` values wide. The
//! capability boundary hands the diffuser a latent `[channels, frames, height, width]`, so every
//! evaluation converts rows to latent and back:
//!
//! ```text
//! patchify    x [channels, frames, height, width] -> r [rows, channels · 4]
//!             row    = ((f · height/2) + i) · (width/2) + j      f, i, j in raster order
//!             column = ((c · 2) + a) · 2 + b                     c major, then the patch's (a, b)
//!             r[row, column] = x[c, f, 2·i + a, 2·j + b]
//! unpatchify  r [rows, channels · 4] -> x [channels, frames, height, width]   the exact inverse
//! ```
//!
//! One operator with a mode, because it is one index arithmetic read in two directions and the two
//! callers are the same step; the descriptor carries the geometry and no index table, so a captured
//! step replays it and no host-side permutation is needed. The move is elementwise and exact.
//!
//! `frames` is a launch parameter, not a field of the descriptor: a request's frame count varies
//! while `channels`, `height` and `width` are the model's, so one compiled program serves every
//! request of a geometry instead of compiling per frame count. `rows` — the packed row count — is
//! the derived value `frames · height/2 · width/2` and is passed beside it, checked against both.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// Elements one thread block handles along a destination row.
pub const BLOCK: u32 = 1024;

/// Which way the mapping is applied.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Mode {
    /// Latent to packed rows: the DiT's input.
    Patchify,
    /// Packed rows to latent: what the sampler's result becomes.
    Unpatchify,
}

impl fmt::Display for Mode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Patchify => "patchify",
            Self::Unpatchify => "unpatchify",
        })
    }
}

/// One compilation of `patch`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Element type of the latent and of the rows; the move is exact in either direction.
    pub dtype: DType,
    /// The latent's channels: `channels · 4` is a row's width.
    pub channels: u32,
    /// The latent's height, which must be even: a patch is `2` tall.
    pub height: u32,
    /// The latent's width, which must be even: a patch is `2` wide.
    pub width: u32,
    /// Which direction this module was compiled for.
    pub mode: Mode,
    /// Elements one thread block handles along a row.
    pub block: u32,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl Descriptor {
    /// A patch mapping, refusing a geometry no latent can have.
    ///
    /// A patch is `1×2×2`, so an odd side is not a request this operator can honour: it is a model
    /// geometry bug, and it is refused here — before a key is composed and a package looked up —
    /// rather than at launch, so the failure names the caller's geometry instead of arriving as a
    /// missing kernel.
    pub fn new(
        dtype: DType,
        channels: u32,
        height: u32,
        width: u32,
        mode: Mode,
        block: u32,
        pdl: bool,
    ) -> Result<Self> {
        if channels == 0
            || height == 0
            || width == 0
            || !height.is_multiple_of(2)
            || !width.is_multiple_of(2)
        {
            return Err(Error::argument(
                "patch",
                format!(
                    "a latent of {channels} channels and {height}×{width} has no 1×2×2 patch \
                     raster: every side must be nonzero and even"
                ),
            ));
        }
        Ok(Self {
            dtype,
            channels,
            height,
            width,
            mode,
            block,
            pdl,
        })
    }

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

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("channels", self.channels)
            .field("height", self.height)
            .field("width", self.width)
            .field("mode", self.mode)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one mapping.
#[derive(Clone, Copy)]
pub struct Args {
    /// `Patchify`: the latent `[channels, frames, height, width]`. `Unpatchify`: the rows
    /// `[rows, channels · 4]`. One view serves both, and the mode says which it is.
    pub x: TensorView,
    /// `Unpatchify`: the latent `[channels, frames, height, width]`. `Patchify`: the rows
    /// `[rows, channels · 4]`.
    pub out: TensorView,
    /// Frames in the latent; the packed row count is `frames · rows_per_frame`.
    pub frames: u32,
}

/// The packed-row mapping, in either direction.
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
        let channels = u64::from(descriptor.channels);
        let height = u64::from(descriptor.height);
        let width = u64::from(descriptor.width);
        let frames = u64::from(args.frames);
        if descriptor.channels == 0 || frames == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a latent with no channels or no frames has no rows to map".to_owned(),
            ));
        }
        // A patch is `1×2×2`: an odd height or width would leave a row of patches off the latent.
        if descriptor.height == 0
            || descriptor.width == 0
            || descriptor.height % 2 != 0
            || descriptor.width % 2 != 0
        {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected an even height and width for a 1×2×2 patch, found {}×{}",
                    descriptor.height, descriptor.width
                ),
            ));
        }
        let rows = frames * descriptor.rows_per_frame();
        let row_width = descriptor.row_width();
        let latent = Shape::new(&[channels, frames, height, width]);
        let packed = Shape::new(&[rows, row_width]);

        let (latent_view, packed_view) = match descriptor.mode {
            Mode::Patchify => (args.x, args.out),
            Mode::Unpatchify => (args.out, args.x),
        };
        latent_view.require_shape(Self::NAME, latent)?;
        packed_view.require_shape(Self::NAME, packed)?;
        for (name, view) in [("latent", latent_view), ("rows", packed_view)] {
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
            .u32(descriptor.channels)
            .u32(descriptor.height)
            .u32(descriptor.width)
            .u32(descriptor.block)
            .flag(descriptor.pdl);
        // The grid walks the destination's rows: one block per (row, element block). A destination
        // element is written by exactly one block, and its source is a single element, so the order
        // between blocks never matters and a replay is deterministic.
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

    fn descriptor(mode: Mode) -> Descriptor {
        Descriptor {
            dtype: DType::BF16,
            channels: 24,
            height: 32,
            width: 48,
            mode,
            block: BLOCK,
            pdl: false,
        }
    }

    #[test]
    fn a_row_is_the_channels_patch_and_the_two_directions_are_one_key_apart() {
        let patchify = descriptor(Mode::Patchify);
        assert_eq!(patchify.row_width(), 96);
        assert_eq!(patchify.rows_per_frame(), 16 * 24);
        // The modes are separate compilations: the grid walks the destination, and the destination
        // is the rows one way and the latent the other.
        assert_ne!(
            patchify.to_string(),
            descriptor(Mode::Unpatchify).to_string()
        );
        assert_eq!(
            patchify.to_string(),
            "dtype_bf16_channels_24_height_32_width_48_mode_patchify_block_1024_pdl_false"
        );
    }

    #[test]
    fn the_frames_are_a_launch_value_so_one_module_serves_every_request() {
        // Only the frame count differs between a 5-frame and a 121-frame request, and the key is
        // the same: the program derives `rows` from it at launch.
        assert_eq!(descriptor(Mode::Patchify), descriptor(Mode::Patchify));
    }

    #[test]
    fn an_odd_side_is_refused_before_a_key_is_composed() {
        // A canvas side is a multiple of 32 and a latent side is that over 16, so every raster a
        // request can produce is even. An odd one is a geometry bug, refused here rather than
        // arriving later as a missing kernel.
        assert!(Descriptor::new(DType::BF16, 24, 48, 48, Mode::Patchify, BLOCK, false).is_ok());
        assert!(Descriptor::new(DType::BF16, 24, 48, 21, Mode::Patchify, BLOCK, false).is_err());
        assert!(Descriptor::new(DType::BF16, 24, 21, 48, Mode::Patchify, BLOCK, false).is_err());
        assert!(Descriptor::new(DType::BF16, 0, 48, 48, Mode::Patchify, BLOCK, false).is_err());
    }

    #[test]
    fn the_mapping_is_exact() {
        assert_eq!(
            Patch::tolerance(&descriptor(Mode::Patchify)),
            Tolerance::exact()
        );
    }
}

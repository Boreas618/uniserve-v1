//! `patchify`: the diffusion transformer's packed rows and its latent's four axes.
//!
//! A diffusion transformer reads **packed rows**: one video row is one `1×2×2` patch of the
//! latent's channels, so a row is `channels · 4` values wide and the rows are the patches in
//! raster order. The capability boundary hands the model a latent `[channels, frames, height,
//! width]`, so every evaluation converts one into the other — and a captured step can only replay
//! a device program, so the mapping is an operator and not host arithmetic:
//!
//! ```text
//! rows = frames · (height / 2) · (width / 2)      columns = channels · 4
//!
//! patchify   rows[row, column] = latent[c, f, 2i + a, 2j + b]
//! unpatchify latent[c, f, 2i + a, 2j + b] = rows[row, column]
//!
//! row    = (f · (height / 2) + i) · (width / 2) + j     raster over frames, then rows, then columns
//! column = (c · 2 + a) · 2 + b                          channel major, then the patch's (a, b)
//! ```
//!
//! The two directions are one operator with a mode, like [`crate::rows`]: they share the geometry,
//! the element type and the element walk, differ only in which side is read, and the mode is
//! compiled in. One thread handles one packed element: the linear index is the row-major index of
//! the packed rows for both modes, so a thread computes the latent offset from it (`patchify`) or
//! reads it (`unpatchify`) — no index table, no host work, and the geometry the descriptor carries
//! is the geometry the launch validates against.
//!
//! Both directions move elements and do no arithmetic, so both are exact.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, Strides, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// The elements one program walks.
pub const PATCH_BLOCK: u32 = 1024;

/// Which side of the mapping a launch writes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Mode {
    /// `rows` receives the latent's patches.
    Pack,
    /// `latent` receives the rows' values.
    Unpack,
}

impl fmt::Display for Mode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Pack => "pack",
            Self::Unpack => "unpack",
        })
    }
}

/// One compilation of `patchify`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Element type of the latent and the rows.
    pub dtype: DType,
    /// Which side is written.
    pub mode: Mode,
    /// Latent channels: a row holds four times this.
    pub channels: u32,
    /// Latent frames.
    pub frames: u32,
    /// Latent rows; even, because a patch is two rows.
    pub height: u32,
    /// Latent columns; even, because a patch is two columns.
    pub width: u32,
    /// Elements one thread block walks.
    pub block: u32,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl Descriptor {
    /// The rows a launch packs, from the geometry.
    pub fn rows(self) -> u64 {
        u64::from(self.frames)
            .saturating_mul(u64::from(self.height / 2))
            .saturating_mul(u64::from(self.width / 2))
    }

    /// The elements one packed row holds.
    pub fn columns(self) -> u64 {
        u64::from(self.channels) * 4
    }
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("mode", self.mode)
            .field("channels", self.channels)
            .field("frames", self.frames)
            .field("height", self.height)
            .field("width", self.width)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one `patchify`.
#[derive(Clone, Copy)]
pub struct Args {
    /// `[channels, frames, height, width]`, contiguous: the latent.
    pub latent: TensorView,
    /// `[rows, columns]`, contiguous: the packed rows.
    pub rows: TensorView,
}

/// The latent ↔ packed-rows mapping of a diffusion transformer.
pub struct Patchify;

impl Op for Patchify {
    const NAME: &'static str = "patchify";

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        if descriptor.channels == 0
            || descriptor.frames == 0
            || descriptor.height == 0
            || descriptor.width == 0
        {
            return Err(Error::argument(
                Self::NAME,
                "a latent has channels, frames, rows and columns".to_owned(),
            ));
        }
        if !descriptor.height.is_multiple_of(2) || !descriptor.width.is_multiple_of(2) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a {0}×{1} latent is not whole 2×2 patches",
                    descriptor.height, descriptor.width
                ),
            ));
        }
        let latent = Shape::new(&[
            u64::from(descriptor.channels),
            u64::from(descriptor.frames),
            u64::from(descriptor.height),
            u64::from(descriptor.width),
        ]);
        args.latent.require_shape(Self::NAME, latent)?;
        if args.latent.dtype != descriptor.dtype {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {} latents, found {}",
                    descriptor.dtype, args.latent.dtype
                ),
            ));
        }
        if args.latent.strides != Strides::row_major(latent) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the latent is read contiguous; found strides {}",
                    args.latent.strides
                ),
            ));
        }

        let rows = descriptor.rows();
        let columns = descriptor.columns();
        args.rows
            .require_shape(Self::NAME, Shape::new(&[rows, columns]))?;
        args.rows
            .require_row(Self::NAME, descriptor.dtype, columns)?;
        if args.rows.strides != Strides::row_major(args.rows.shape) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the rows are read contiguous; found strides {}",
                    args.rows.strides
                ),
            ));
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.latent)
            .view(args.rows)
            .u32(descriptor.channels)
            .u32(descriptor.frames)
            .u32(descriptor.height)
            .u32(descriptor.width)
            .u32(descriptor.block)
            // One program serves both directions; the mode is compiled in, and the flag travels so
            // a package that reads it and one that bakes it agree on the same argument list.
            .u32(match descriptor.mode {
                Mode::Pack => 0,
                Mode::Unpack => 1,
            })
            .flag(descriptor.pdl);
        kernel.launch(
            LaunchGrid::covering(rows * columns, u64::from(descriptor.block))?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        Args {
            rows: twins.shadow_of(args.rows),
            latent: twins.shadow_of(args.latent),
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // A move: every element arrives unchanged in either direction.
        Tolerance::exact()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(mode: Mode) -> Descriptor {
        Descriptor {
            dtype: DType::BF16,
            mode,
            channels: 24,
            frames: 4,
            height: 16,
            width: 16,
            block: PATCH_BLOCK,
            pdl: false,
        }
    }

    #[test]
    fn the_geometry_names_the_rows_and_their_width() {
        let geometry = descriptor(Mode::Pack);
        assert_eq!(geometry.rows(), 4 * 8 * 8);
        assert_eq!(geometry.columns(), 96);
        assert_eq!(
            geometry.rows() * geometry.columns(),
            24 * 4 * 16 * 16,
            "the two sides hold the same elements"
        );
    }

    #[test]
    fn the_key_states_the_mode_and_the_four_extents() {
        assert_eq!(
            descriptor(Mode::Pack).to_string(),
            "dtype_bf16_mode_pack_channels_24_frames_4_height_16_width_16_block_1024_pdl_false"
        );
        assert_eq!(
            descriptor(Mode::Unpack).to_string(),
            "dtype_bf16_mode_unpack_channels_24_frames_4_height_16_width_16_block_1024_pdl_false"
        );
    }

    #[test]
    fn a_move_is_exact_in_both_directions() {
        assert_eq!(Patchify::tolerance(&descriptor(Mode::Pack)).atol, 0.0);
        assert_eq!(Patchify::tolerance(&descriptor(Mode::Unpack)).atol, 0.0);
        assert_eq!(Patchify::tolerance(&descriptor(Mode::Pack)).rtol, 0.0);
    }
}

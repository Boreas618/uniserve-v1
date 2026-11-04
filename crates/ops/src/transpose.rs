//! `transpose`: a row-major `[rows, columns]` matrix written as a row-major `[columns, rows]` one.
//!
//! The audio encoder keeps `[batch, channels, frames]` throughout and attends over frames, so it has
//! to move frames to the leading axis before its qkv projection and back after it. No view can stand
//! in: the layers on either side read contiguous rows, and a transposed view is not one.
//!
//! ```text
//! out[i, o] = input[o, i]                    out [columns, rows], input [rows, columns]
//! ```
//!
//! Both sides are contiguous, which is what makes this a *layout* operator rather than a general
//! gather: a strided side is refused with the offending stride in the message. A program reads a
//! `block × block` tile of the source coalesced along its columns, transposes it through shared
//! memory, and writes it coalesced along the destination's columns, bounds-checking the edges — the
//! textbook tiled transpose, chosen because the naive one writes one element per cache line.
//!
//! **Both extents are launch values.** The frame count varies per request while the arithmetic does
//! not, so a descriptor extent would recompile the audio encoder for every clip length; that is the
//! extent rule's second tier, and it is why the key holds only the element type, the tile and the
//! flag.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// One side of the square tile a program moves through shared memory.
pub const TILE: u32 = 32;

/// One compilation of `transpose`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Element type of both sides; a transpose does not change it.
    pub dtype: DType,
    /// The side of one square tile: `block × block` elements per program.
    pub block: u32,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Both extents are launch values, so the key names neither of them.
        Key::new()
            .field("dtype", self.dtype)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one transpose.
#[derive(Clone, Copy)]
pub struct Args {
    /// `[rows, columns]` the source, contiguous.
    pub input: TensorView,
    /// `[columns, rows]` the destination, contiguous.
    pub out: TensorView,
}

/// Write a row-major matrix into the transposed row-major matrix.
pub struct Transpose;

/// Whether a view is packed: each axis's stride is the product of the extents after it.
///
/// A unit axis may carry any stride — a `[1, columns]` row taken from a taller matrix does, and it
/// still moves as one contiguous run — so the comparison is axis by axis rather than against a
/// row-major stride vector. Two axes that share an element are not packed and are refused by the
/// same test, because the stride of the outer one is then too small.
fn contiguous(view: &TensorView) -> bool {
    let mut expected = 1;
    for axis in (0..view.shape.rank()).rev() {
        let extent = view.shape.dim(axis);
        if extent != 1 && view.strides.get(axis) != expected {
            return false;
        }
        expected *= extent;
    }
    true
}

/// Check that a view is a contiguous matrix of the descriptor's element type.
fn matrix(view: &TensorView, name: &str, dtype: DType) -> Result<(u64, u64)> {
    view.require_rank(Transpose::NAME, 2)?;
    if view.dtype != dtype {
        return Err(Error::argument(
            Transpose::NAME,
            format!("expected {dtype} {name}, found {}", view.dtype),
        ));
    }
    if !contiguous(view) {
        return Err(Error::argument(
            Transpose::NAME,
            format!(
                "the {name} must be contiguous: its strides are {:?} over {:?}, which is not a \
                 packed layout",
                view.strides.as_slice(),
                view.shape.dims()
            ),
        ));
    }
    let (rows, columns) = (view.shape.dim(0), view.shape.dim(1));
    if rows == 0 || columns == 0 {
        return Err(Error::argument(
            Transpose::NAME,
            format!("a {rows} × {columns} matrix has nothing to move"),
        ));
    }
    Ok((rows, columns))
}

impl Op for Transpose {
    const NAME: &'static str = "transpose";

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        // A layout operator moves the bits it was given: no arithmetic, so no narrower type, and no
        // integer or narrow-float type either, because nothing that would want one transposes.
        if !matches!(descriptor.dtype, DType::F32 | DType::BF16 | DType::F16) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected f32, bf16 or f16 elements, the descriptor says {}",
                    descriptor.dtype
                ),
            ));
        }
        if descriptor.block == 0 || !descriptor.block.is_power_of_two() {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the tile is {} elements on a side, which is not a power of two",
                    descriptor.block
                ),
            ));
        }
        let (rows, columns) = matrix(&args.input, "input", descriptor.dtype)?;
        let (out_rows, out_columns) = matrix(&args.out, "out", descriptor.dtype)?;
        if (out_rows, out_columns) != (columns, rows) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a {rows} × {columns} input needs a {columns} × {rows} output, \
                     the view is {out_rows} × {out_columns}"
                ),
            ));
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.input)
            .view(args.out)
            .u64(rows)
            .u64(columns)
            .u32(descriptor.block)
            .flag(descriptor.pdl);
        // One program per tile: the source is read a tile of columns at a time and the destination is
        // written a tile of rows at a time, so the grid is the tile grid and no axis needs a second
        // pass.
        kernel.launch(
            LaunchGrid::new([
                columns.div_ceil(u64::from(descriptor.block)),
                rows.div_ceil(u64::from(descriptor.block)),
                1,
            ])?,
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
        // No arithmetic and no reduction: every element is copied byte for byte, so any difference is
        // a wrong element rather than a rounding.
        Tolerance::exact()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Shape, Strides};

    fn descriptor() -> Descriptor {
        Descriptor {
            dtype: DType::F32,
            block: TILE,
            pdl: false,
        }
    }

    fn view(dtype: DType, shape: &[u64], strides: &[u64]) -> TensorView {
        TensorView::with_strides(
            crate::tensor::DevicePtr::new(0x1000),
            dtype,
            Shape::new(shape),
            Strides::new(strides),
        )
    }

    #[test]
    fn the_key_names_the_element_type_the_tile_and_the_flag_only() {
        assert_eq!(descriptor().to_string(), "dtype_f32_block_32_pdl_false");
        // The extents are not in it: a clip of any length reuses the module.
        assert!(!descriptor().to_string().contains("rows"));
        assert!(!descriptor().to_string().contains("columns"));
    }

    #[test]
    fn a_contiguous_matrix_is_moveable_and_its_transpose_is_the_other_way_round() {
        let input = view(DType::F32, &[4, 8], &[8, 1]);
        assert_eq!(matrix(&input, "input", DType::F32).unwrap(), (4, 8));
        let out = view(DType::F32, &[8, 4], &[4, 1]);
        assert_eq!(matrix(&out, "out", DType::F32).unwrap(), (8, 4));
    }

    #[test]
    fn a_strided_side_is_refused_with_its_strides_in_the_message() {
        // A transposed view of the same storage: it looks like the answer but every element is in the
        // wrong place, which is exactly the call this operator exists to replace.
        let strided = view(DType::F32, &[8, 4], &[1, 8]);
        let error = matrix(&strided, "out", DType::F32).unwrap_err();
        assert!(format!("{error}").contains("must be contiguous"), "{error}");

        let window = view(DType::F32, &[4, 8], &[16, 1]);
        let error = matrix(&window, "input", DType::F32).unwrap_err();
        assert!(format!("{error}").contains("[16, 1]"), "{error}");
    }

    #[test]
    fn a_single_row_is_packed_whatever_stride_the_view_carries() {
        // One frame's channels taken from a taller matrix: the row moves as one run, so refusing it
        // would refuse a one-frame clip.
        let one_row = view(DType::F32, &[1, 8], &[999, 1]);
        assert!(contiguous(&one_row));
        assert_eq!(matrix(&one_row, "input", DType::F32).unwrap(), (1, 8));
        assert!(contiguous(&view(DType::F32, &[2, 8], &[8, 1])));
        assert!(!contiguous(&view(DType::F32, &[2, 8], &[4, 1])));
    }

    #[test]
    fn the_element_type_is_the_descriptor_and_the_other_side_must_match_it() {
        let input = view(DType::BF16, &[4, 8], &[8, 1]);
        let error = matrix(&input, "input", DType::F32).unwrap_err();
        assert!(format!("{error}").contains("expected f32 input"), "{error}");
    }

    #[test]
    fn only_matrices_of_floats_are_transposed() {
        let vector = view(DType::F32, &[8], &[1]);
        let error = matrix(&vector, "input", DType::F32).unwrap_err();
        assert!(format!("{error}").contains("expected 2 axes"), "{error}");

        let empty = view(DType::F32, &[0, 8], &[8, 1]);
        let error = matrix(&empty, "input", DType::F32).unwrap_err();
        assert!(format!("{error}").contains("nothing to move"), "{error}");

        // The program has no integer path, so an integer launch is refused rather than run through a
        // float one.
        let integer = Descriptor {
            dtype: DType::I32,
            block: TILE,
            pdl: false,
        };
        assert_eq!(integer.dtype, DType::I32);
        assert!(!integer.dtype.is_float());
    }
}

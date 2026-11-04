//! `copy_block`: moving a logical block between two views of different layout.
//!
//! ```text
//! out[c, f, h, w] = source[c, f, h, w]      for every index of the block
//! ```
//!
//! Each side is addressed through its own strides, so one program copies, places a block inside a
//! larger buffer, and permutes axes — a permuted copy is the same kernel with the source's stride
//! vector in the destination's axis order. The destination's strides are **launch scalars** and not
//! descriptor fields: a clip's tile geometry varies per request, and a key that named the strides
//! would compile one program per canvas, so a captured call would stop being the same program on the
//! next one.
//!
//! The rank is the block's, four: `[channels, frames, height, width]` for a video latent and
//! `[frames, height, width, channels]` for the raster the media stage reads. A model with a batch axis
//! folds it into the first, which is the caller's business and not a second compilation.
//!
//! Two rules this operator deliberately does **not** have:
//!
//! - **No permutation field.** A permutation is the source's strides permuted; a mode would be a
//!   second way to say the same thing, and the two could disagree.
//! - **No unit-stride rule on the destination.** `crate::tensor::row_writable` exists for operators
//!   that store one contiguous row per program; this one walks whatever strides it is given, so a
//!   destination whose innermost axis is not unit stride is legal.
//!
//! There is no arithmetic: the same bits arrive on the other side, which is what makes a reorder
//! bit-exact and the tolerance [`Tolerance::exact`].

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// Elements one program moves.
pub const COPY: u32 = 1024;

/// The rank of every block this operator moves.
pub const RANK: usize = 4;

/// One compilation of `copy_block`.
///
/// The two element types are in the key even though a copy moves bits between views of one type: a
/// future converting copy keys differently instead of silently converting under this one, and a
/// launch whose views disagree with the descriptor is refused by name.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Element type of the source view.
    pub source: DType,
    /// Element type of the destination view.
    pub out: DType,
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("source", self.source)
            .field("out", self.out)
            .finish(formatter)
    }
}

/// The arguments of one `copy_block`.
#[derive(Clone, Copy)]
pub struct Args<'a> {
    /// The source view, read through its own strides.
    pub source: TensorView,
    /// The destination view, whose extents are the block's.
    pub out: TensorView,
    /// The destination's four strides, in the block's axis order.
    ///
    /// A launch value rather than a descriptor field, and stated by the caller rather than read off
    /// `out`, because a caller may address a region of a buffer whose view it built row-major: the
    /// strides say where each element lands, and a view's default strides would not.
    pub out_strides: &'a [i64; RANK],
}

/// The alignment a view's strides must have to be moved element by element.
///
/// A stride is in elements and counted along that view's own axes; a negative stride is refused here
/// rather than in a kernel that would read before its buffer.
fn check_strides(name: &'static str, strides: &[i64; RANK], extents: &[u64; RANK]) -> Result<()> {
    for axis in 0..RANK {
        if extents[axis] == 0 {
            continue;
        }
        if strides[axis] < 0 {
            return Err(Error::argument(
                name,
                format!(
                    "axis {axis} has stride {}: a block copy reads and writes forward",
                    strides[axis]
                ),
            ));
        }
    }
    Ok(())
}

/// One block of one view into another.
pub struct CopyBlock;

impl Op for CopyBlock {
    const NAME: &'static str = "copy_block";

    type Descriptor = Descriptor;
    type Args<'a> = Args<'a>;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        if descriptor.source != descriptor.out {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a copy of {} into {} converts nothing: both sides move {} elements",
                    descriptor.source, descriptor.out, descriptor.source
                ),
            ));
        }
        for (name, view, dtype) in [
            ("source", args.source, descriptor.source),
            ("out", args.out, descriptor.out),
        ] {
            if view.dtype != dtype {
                return Err(Error::argument(
                    Self::NAME,
                    format!("expected {dtype} {name}, found {}", view.dtype),
                ));
            }
            view.require_rank(Self::NAME, RANK)?;
        }
        let extents = [
            args.out.shape.dim(0),
            args.out.shape.dim(1),
            args.out.shape.dim(2),
            args.out.shape.dim(3),
        ];
        for (axis, extent) in extents.iter().enumerate() {
            if args.source.shape.dim(axis) != *extent {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "axis {axis}: the source has {} elements and the destination {extent}; a \
                         block copy moves one logical block",
                        args.source.shape.dim(axis)
                    ),
                ));
            }
        }
        if extents.contains(&0) {
            return Ok(());
        }
        let strides = args.source.strides.as_slice();
        let source_strides = [strides[0], strides[1], strides[2], strides[3]];
        let source_strides = [
            i64::try_from(source_strides[0]).map_err(strided("source", 0))?,
            i64::try_from(source_strides[1]).map_err(strided("source", 1))?,
            i64::try_from(source_strides[2]).map_err(strided("source", 2))?,
            i64::try_from(source_strides[3]).map_err(strided("source", 3))?,
        ];
        check_strides(Self::NAME, &source_strides, &extents)?;
        check_strides(Self::NAME, args.out_strides, &extents)?;

        let mut arguments = Arguments::new();
        arguments.view(args.source).view(args.out);
        for extent in extents {
            arguments.u64(extent);
        }
        for stride in source_strides.iter().chain(args.out_strides.iter()) {
            arguments.i64(*stride);
        }
        let elements = extents.iter().product::<u64>();
        // `total`, the elements one program moves, and the PDL flag -- and nothing after them. The
        // block width is NOT the last word a module takes: it is a `tl.constexpr`, fixed by the
        // module's launch options and never passed in the argument list. A trailing block word here
        // is one parameter more than the kernel has, which reaches the driver as an unnamed argument
        // rather than as a refusal.
        arguments.u64(elements).u32(COPY).flag(false);
        kernel.launch(
            LaunchGrid::covering(elements, u64::from(COPY))?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        // The destination is the twin: a twin of the source would compare the source with itself and
        // could never fail (specs/worker.md 7.3 — the mistake `ops::rows` had).
        Args {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // No arithmetic: the same bits arrive on the other side.
        Tolerance::exact()
    }
}

/// The failure of one side's stride not fitting the signed 64-bit launch scalar.
fn strided<'a>(
    side: &'static str,
    axis: usize,
) -> impl Fn(std::num::TryFromIntError) -> Error + 'a {
    move |_| {
        Error::argument(
            CopyBlock::NAME,
            format!("the {side} stride of axis {axis} does not fit a signed 64-bit parameter"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_strides_are_not_part_of_the_compilation() {
        // A captured launch must stay valid when a canvas's tile geometry changes, which is the whole
        // reason the strides are arguments: the key names the two element types and nothing else.
        let plain = Descriptor {
            source: DType::BF16,
            out: DType::BF16,
        };
        assert_eq!(crate::descriptor::key(&plain), "source_bf16_out_bf16");
        assert_ne!(
            crate::descriptor::key(&plain),
            crate::descriptor::key(&Descriptor {
                source: DType::F32,
                out: DType::F32
            })
        );
        assert_ne!(
            crate::descriptor::key(&plain),
            crate::descriptor::key(&Descriptor {
                source: DType::BF16,
                out: DType::F32
            })
        );
    }
}

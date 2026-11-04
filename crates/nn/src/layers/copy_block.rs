//! Moving a block between two views of different layout (`ops::copy_block`).
//!
//! The video decoder writes a latent channel-first, `[channels, frames, height, width]`, and the
//! media stage reads bytes channel-last, `[frames, height, width, channels]`: the same block, two
//! layouts. One strided copy moves it, and the permutation lives in the strides rather than in a
//! field, so this layer has one compiled form for every geometry a request can have.
//!
//! The layer carries no weights and holds no state beyond its element type: the two views are the
//! caller's, and the destination's strides travel with the call exactly as the operator takes them,
//! because a clip's tile shape changes per request while the program does not.

use ops::DType;
use ops::TensorView;
use ops::copy_block::{self, CopyBlock as CopyBlockOp, RANK};

use crate::error::{Error, Result};
use crate::forward::Forward;

/// A block of `dtype` elements moved between two four-axis views.
#[derive(Clone, Copy)]
pub struct CopyBlock {
    dtype: DType,
}

impl CopyBlock {
    /// A copy of four-axis blocks of `dtype`.
    pub fn new(dtype: DType) -> Result<Self> {
        if dtype.bytes().is_none() {
            return Err(Error::shape(
                "CopyBlock",
                format!("{dtype} elements are sub-byte; a block copy moves whole elements"),
            ));
        }
        Ok(Self { dtype })
    }

    /// `destination[c, f, h, w] = source[c, f, h, w]` under each side's own strides.
    ///
    /// The destination's strides are passed rather than read off its view: a caller may address a
    /// region of a larger raster whose view it built row-major, and the strides are what say where
    /// each element lands. Both views must be four-axis and hold the same extents.
    pub fn forward(
        &self,
        f: &Forward,
        source: TensorView,
        destination: TensorView,
        destination_strides: &[i64; RANK],
    ) -> Result<()> {
        for (name, view) in [("source", source), ("destination", destination)] {
            if view.shape.rank() != RANK {
                return Err(Error::shape(
                    "CopyBlock",
                    format!(
                        "the {name} view has {} axes; a block copy moves four and a batch axis is \
                         folded into the first",
                        view.shape.rank()
                    ),
                ));
            }
        }
        let descriptor = copy_block::Descriptor {
            source: self.dtype,
            out: self.dtype,
        };
        f.operate::<CopyBlockOp>(
            &descriptor,
            copy_block::Args {
                source,
                out: destination,
                out_strides: destination_strides,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sub_byte_element_type_is_refused() {
        assert!(CopyBlock::new(DType::BF16).is_ok());
        assert!(CopyBlock::new(DType::F32).is_ok());
        assert!(CopyBlock::new(DType::F4E2M1).is_err());
    }
}

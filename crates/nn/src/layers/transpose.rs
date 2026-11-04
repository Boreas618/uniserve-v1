//! The matrix transpose a model composes when a layer's rows are the wrong way round.
//!
//! An audio encoder's signal is `[batch, channels, frames]` and it attends over frames, while the
//! layers after that (`Linear`, `LayerNorm`) read contiguous rows — so the `channels × frames` plane
//! has to become `frames × channels` in memory, not merely in a view. A permuted view cannot stand
//! in: the operator writes a new buffer through shared memory tiles, and this layer is the call.
//!
//! Both extents are **launch values**, so one compiled program serves every clip length: the
//! descriptor holds the element type and the tile side, and nothing of the request's geometry.

use ops::transpose::{self, Transpose as TransposeOp};
use ops::{DType, TensorView};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::launch::PDL;

/// The square tile one program moves through shared memory.
const BLOCK: u32 = transpose::TILE;

/// A row-major matrix written into its transposed row-major form.
#[derive(Clone, Copy)]
pub struct Transpose {
    dtype: DType,
}

impl Transpose {
    /// The transpose of a `dtype` matrix.
    pub fn new(dtype: DType) -> Result<Self> {
        if dtype.bytes().is_none() {
            return Err(Error::shape(
                "Transpose",
                format!("{dtype} has no whole-byte element size to move"),
            ));
        }
        Ok(Self { dtype })
    }

    /// The matrix element type.
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Write `source [rows, columns]` into `out [columns, rows]`.
    ///
    /// Both views are contiguous row-major matrices of the layer's type. A caller that has a batch
    /// axis folds it into the rows — `[1, channels, frames]` is the `[channels, frames]` matrix this
    /// takes, because the batch is the caller's and not the transpose's. A rank that is not two is
    /// refused with the shape in the message rather than reinterpreted.
    pub fn forward(&self, f: &Forward, source: TensorView, out: TensorView) -> Result<()> {
        for (name, view) in [("source", source), ("out", out)] {
            if view.shape.rank() != 2 {
                return Err(Error::shape(
                    "Transpose",
                    format!(
                        "the {name} is {:?}, a [rows, columns] matrix expected: a batch axis is \
                         the caller's to fold into the rows",
                        view.shape.dims()
                    ),
                ));
            }
            if view.dtype != self.dtype {
                return Err(Error::shape(
                    "Transpose",
                    format!(
                        "the {name} is {}, the layer was built for {}",
                        view.dtype, self.dtype
                    ),
                ));
            }
        }
        let (rows, columns) = (source.shape.dim(0), source.shape.dim(1));
        if out.shape.dims() != [columns, rows] {
            return Err(Error::shape(
                "Transpose",
                format!(
                    "{:?} is not the transpose of {:?}",
                    out.shape.dims(),
                    source.shape.dims()
                ),
            ));
        }
        let descriptor = transpose::Descriptor {
            dtype: self.dtype,
            block: BLOCK,
            pdl: PDL,
        };
        f.operate::<TransposeOp>(&descriptor, transpose::Args { input: source, out })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_transpose_is_typed_by_its_element_type_alone() {
        assert!(Transpose::new(DType::BF16).is_ok());
        assert!(Transpose::new(DType::F32).is_ok());
        assert!(Transpose::new(DType::U8).is_ok());
    }
}

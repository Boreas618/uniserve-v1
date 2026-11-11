//! The matrix transpose a model composes when a layer's rows are the wrong way round.
//!
//! An audio encoder's signal is `[batch, channels, frames]` and it attends over frames, while the
//! layers after that (`Linear`, `LayerNorm`) read contiguous rows — so the `channels × frames` plane
//! has to become `frames × channels` in memory, not merely in a view. A permuted view cannot stand
//! in: the operator writes a new buffer through shared memory tiles, and this layer is the call.
//!
//! Both extents are **launch values**, so one compiled program serves every clip length: the
//! descriptor holds the element type and the tile side, and nothing of the request's geometry.

use ops::Operand;
use ops::transpose::{self, Transpose as TransposeOp};
use uniserve_tensor::DType;

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
        if ops::abi::bytes(dtype).is_none() {
            return Err(Error::shape(
                "Transpose",
                format!(
                    "{dtype} has no whole-byte element size to move",
                    dtype = ops::abi::name(dtype)
                ),
            ));
        }
        Ok(Self { dtype })
    }

    /// The descriptor this layer's launch takes.
    ///
    /// No caller's element type: a transpose is typed by the matrix it was built for, and both
    /// operands are that type.
    fn descriptor(&self) -> transpose::Descriptor {
        transpose::Descriptor {
            dtype: self.dtype,
            block: BLOCK,
            pdl: PDL,
        }
    }

    /// `source [rows, columns]` transposed into a fresh `[columns, rows]` matrix.
    ///
    /// `source` is a contiguous row-major matrix of the layer's type. A caller that has a batch
    /// axis folds it into the rows — `[1, channels, frames]` is the `[channels, frames]` matrix this
    /// takes, because the batch is the caller's and not the transpose's. A rank that is not two is
    /// refused with the shape in the message rather than reinterpreted.
    ///
    /// # Errors
    ///
    /// A source of another rank or element type, an arena that cannot hold the output, or the
    /// operator's refusal.
    pub fn forward(&self, f: &Forward, source: Operand) -> Result<Operand> {
        if source.shape.rank() != 2 {
            return Err(Error::shape(
                "Transpose",
                format!(
                    "the source is {:?}, a [rows, columns] matrix expected: a batch axis is the \
                     caller's to fold into the rows",
                    source.shape.dims()
                ),
            ));
        }
        if source.dtype != self.dtype {
            return Err(Error::shape(
                "Transpose",
                format!(
                    "the source is {}, the layer was built for {}",
                    ops::abi::name(source.dtype),
                    ops::abi::name(self.dtype)
                ),
            ));
        }
        let (rows, columns) = (source.shape.dim(0), source.shape.dim(1));
        let out = f.empty(self.dtype, ops::Shape::new(&[columns, rows]))?;
        let descriptor = self.descriptor();
        f.operate::<TransposeOp>(&descriptor, transpose::Args { input: source, out })?;
        Ok(out)
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

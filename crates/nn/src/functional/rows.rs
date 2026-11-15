//! Placing rows a batch's tokens leave in memory execution lends: per-request state that is
//! written by row and not by slot of a recurrent operator.
//!
//! ```text
//! destination[indices[j]] = source[j]        for indices[j] ≥ 0; a negative index writes nothing
//! ```
//!
//! A token that leaves no row, and a padding token of a bucket wider than its batch, state a
//! negative index: [`NO_SLOT`](crate::NO_SLOT) read as I32. The rows a launch names are
//! distinct; two tokens that name one row leave either's.

use ops::rows::{Args, Descriptor, Mode, Rows};
use tensor::{DType, Device, Tensor};

use super::{Fused, compute};
use crate::error::{Error, Result};

/// Writes the rows of `source`, `[count, columns]`, into `destination`, `[rows, columns]` of
/// the same element type, in place: source row `j` at destination row `indices[j]`, an I32
/// column, and nowhere when the index is negative.
///
/// # Errors
///
/// Rows of another width or element type than the destination's, indices that are not one I32
/// per source row, a destination that is a strided view, or the operator's refusal.
pub fn place_rows_(destination: &Tensor, indices: &Tensor, source: &Tensor) -> Result<()> {
    const OWNER: &str = PlaceRowsCall::OWNER;

    let (&[_, columns], &[count, width]) = (destination.dims(), source.dims()) else {
        return Err(Error::shape(
            OWNER,
            format!(
                "a destination of {:?} and rows of {:?}: [rows, columns] and [count, columns] \
                 expected",
                destination.dims(),
                source.dims()
            ),
        ));
    };
    if width != columns
        || source.dtype() != destination.dtype()
        || indices.dims() != [count]
        || indices.dtype() != DType::I32
    {
        return Err(Error::shape(
            OWNER,
            format!(
                "{} rows of {:?} at {} indices of {:?} into {} rows of {columns}",
                ops::abi::name(source.dtype()),
                source.dims(),
                ops::abi::name(indices.dtype()),
                indices.dims(),
                ops::abi::name(destination.dtype()),
            ),
        ));
    }

    compute(PlaceRowsCall {
        destination,
        indices,
        source,
        columns,
    })
}

/// A checked call of [`place_rows_`]: rows of `columns` elements.
struct PlaceRowsCall<'t> {
    destination: &'t Tensor,
    indices: &'t Tensor,
    source: &'t Tensor,
    columns: usize,
}

impl PlaceRowsCall<'_> {
    const OWNER: &'static str = "place_rows";
}

impl Fused for PlaceRowsCall<'_> {
    type Output = ();

    fn device(&self) -> &Device {
        self.destination.device()
    }

    fn cpu(self) -> Result<()> {
        let Self {
            destination,
            indices,
            source,
            ..
        } = self;
        for (row, index) in indices.to_vec1::<i32>()?.into_iter().enumerate() {
            if index >= 0 {
                destination.slice_set(&source.narrow(0, row, 1)?, 0, index as usize)?;
            }
        }
        Ok(())
    }

    fn launch(self) -> Result<()> {
        let Self {
            destination,
            indices,
            source,
            columns,
        } = self;
        ops::launch::<Rows>(
            &Descriptor::new(destination.dtype(), Mode::Place, columns as u32),
            Args {
                source: ops::operand(source)?,
                destination: ops::operand(destination)?,
                indices: ops::operand(indices)?,
            },
        )?;
        Ok(())
    }
}

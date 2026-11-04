//! Placing the rows a tower contributes into the sequence that reserved them.
//!
//! Two modalities of the H3 text encoder funnel rows into a packed sequence (`ops::rows`):
//!
//! - a tower's vision rows *replace* the placeholder rows the prompt reserved for its images, so the
//!   text encoder attends them in the position the prompt put them;
//! - a deep-stack tap *adds* a tower layer's rows into rows that already hold a residual, which is
//!   how the encoder folds an intermediate feature into its input stream.
//!
//! The layer carries no weights: the destination, the source and the index view are the caller's,
//! because the sequence and its placeholders belong to the encoder's composition. It owns only the
//! compiled form — one operator variant per mode, width and element block — so a block that places
//! rows twice pays for one program.

use ops::DType;
use ops::TensorView;
use ops::rows::{self, Mode, Rows as RowsOp};

use crate::error::{Error, Result};
use crate::forward::Forward;

/// Elements one thread block places.
const BLOCK: u32 = 1024;

/// Row placement over a `[rows, columns]` sequence.
#[derive(Clone, Copy)]
pub struct Rows {
    datatype: DType,
    columns: u32,
}

impl Rows {
    /// A layer that places rows of a sequence `columns` wide, both of `dtype`.
    pub fn new(dtype: DType, columns: u64) -> Result<Self> {
        let columns = u32::try_from(columns).map_err(|_| {
            Error::shape(
                "Rows",
                format!("a row of {columns} elements is wider than a descriptor holds"),
            )
        })?;
        Ok(Self {
            datatype: dtype,
            columns,
        })
    }

    /// `destination[indices[j]] = source[j]`: the vision rows take their placeholders' place.
    pub fn replace(
        &self,
        f: &Forward,
        destination: TensorView,
        source: TensorView,
        indices: TensorView,
    ) -> Result<()> {
        self.forward(f, Mode::Replace, destination, source, indices)
    }

    /// `destination[indices[j]] += source[j]`: a tap adds into the rows already there.
    pub fn add(
        &self,
        f: &Forward,
        destination: TensorView,
        source: TensorView,
        indices: TensorView,
    ) -> Result<()> {
        self.forward(f, Mode::Add, destination, source, indices)
    }

    fn forward(
        &self,
        f: &Forward,
        mode: Mode,
        destination: TensorView,
        source: TensorView,
        indices: TensorView,
    ) -> Result<()> {
        let descriptor = rows::Descriptor {
            dtype: self.datatype,
            mode,
            columns: self.columns,
            block: BLOCK,
            pdl: false,
        };
        f.operate::<RowsOp>(
            &descriptor,
            rows::Args {
                source,
                destination,
                indices,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_width_wider_than_a_descriptor_holds_is_refused() {
        assert!(Rows::new(DType::BF16, 5120).is_ok());
        assert!(Rows::new(DType::BF16, u64::from(u32::MAX) + 1).is_err());
    }
}

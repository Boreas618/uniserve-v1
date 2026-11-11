//! Row indexing: rows picked out of a `[rows, columns]` sequence by index, and rows written into one.
//!
//! ```text
//! IndexRows::gather        out[j] = source[indices[j]]                 a fresh tensor
//! IndexRows::scatter       destination[indices[j]] = source[j]
//! IndexRows::scatter_add   destination[indices[j]] += source[j]      one rounding into destination's type
//! ```
//!
//! A gather and a scatter move whole rows and are exact. `indices` is a flat `[count]` int32 view
//! and may repeat: a scatter writes a repeated index more than once in the order it appears, so the
//! caller controls a sum's order and repeated launches agree bit for bit. The layer holds nothing:
//! a gather reserves its rows from the forward's arena, a scatter writes the caller's destination
//! in place, and the element type and the row width are read from the views.
//!
//! # The caller's obligation
//!
//! Every index must be less than the indexed view's row count. The kernels take an index as an
//! address and mask only the columns, so an index computed for another canvas — a clip's row table
//! launched against a tile's rows — reads or writes whatever the arena put next instead of
//! refusing. A device value cannot be checked on the host, so the table's provenance is the
//! caller's to guarantee: it must be computed for the same canvas as the view it is launched
//! against.

use ops::logits::{self, LogitsGather};
use ops::rows::{self, Mode, Rows};
use ops::{Operand, Shape};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::launch::PDL;

/// Elements one thread block moves.
const BLOCK: u32 = 1024;

/// The width of `view`'s rows as a descriptor field.
fn columns(owner: &'static str, view: &Operand) -> Result<u32> {
    u32::try_from(view.last()).map_err(|_| {
        Error::shape(
            owner,
            format!(
                "a row of {} elements is wider than a descriptor holds",
                view.last()
            ),
        )
    })
}

/// A flat `[count]` index view: the operators derive the row count from it and refuse more axes.
fn flat(owner: &'static str, indices: &Operand) -> Result<u64> {
    if indices.shape.rank() != 1 {
        return Err(Error::shape(
            owner,
            format!(
                "the indices are {:?}, and the operator takes a flat [count] index view",
                indices.shape.dims()
            ),
        ));
    }
    Ok(indices.numel())
}

/// Rows of a sequence read and written by index.
#[derive(Clone, Copy, Debug)]
pub struct IndexRows;

impl IndexRows {
    /// `source[indices[j]]` for each index: a fresh `[count, columns]` tensor of `source`'s
    /// element type.
    ///
    /// # Errors
    ///
    /// Indices of another rank, an arena that cannot hold the output, or the operator's refusal.
    pub fn gather(&self, f: &Forward, source: Operand, indices: Operand) -> Result<Operand> {
        const OWNER: &str = "IndexRows::gather";
        let count = flat(OWNER, &indices)?;
        let out = f.empty(source.dtype, Shape::new(&[count, source.last()]))?;
        let descriptor = logits::GatherDescriptor {
            dtype: source.dtype,
            columns: columns(OWNER, &source)?,
            block: BLOCK,
            pdl: PDL,
        };
        let args = logits::GatherArgs {
            hidden: source,
            rows: indices,
            out,
        };
        f.operate::<LogitsGather>(&descriptor, args)?;
        Ok(out)
    }

    /// `destination[indices[j]] = source[j]`, in place on `destination`.
    ///
    /// # Errors
    ///
    /// The operator's refusal of the three views' shapes.
    pub fn scatter(
        &self,
        f: &Forward,
        source: Operand,
        indices: Operand,
        destination: Operand,
    ) -> Result<()> {
        Self::rows(f, Mode::Replace, source, indices, destination)
    }

    /// `destination[indices[j]] += source[j]`, in place on `destination`.
    ///
    /// # Errors
    ///
    /// The operator's refusal of the three views' shapes.
    pub fn scatter_add(
        &self,
        f: &Forward,
        source: Operand,
        indices: Operand,
        destination: Operand,
    ) -> Result<()> {
        Self::rows(f, Mode::Add, source, indices, destination)
    }

    /// The `rows` launch of `mode`: a replace and an add are two programs over the same table.
    fn rows(
        f: &Forward,
        mode: Mode,
        source: Operand,
        indices: Operand,
        destination: Operand,
    ) -> Result<()> {
        let descriptor = rows::Descriptor {
            dtype: destination.dtype,
            mode,
            columns: columns("IndexRows", &destination)?,
            block: BLOCK,
            pdl: false,
        };
        let args = rows::Args {
            source,
            destination,
            indices,
        };
        f.operate::<Rows>(&descriptor, args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Arena;
    use uniserve_tensor::DType;

    /// A gather reads a flat index view and writes one row per index of the source's width.
    #[test]
    fn a_gather_writes_one_source_row_per_index() {
        let arena = Arena::counting();
        let f = Forward::recording(&arena);
        let view = |dtype, dims: &[u64]| arena.tensor(dtype, Shape::new(dims)).unwrap();
        let source = view(DType::BF16, &[8, 32]);
        let gathered = IndexRows
            .gather(&f, source, view(DType::I32, &[3]))
            .unwrap();
        assert_eq!(gathered.shape.dims(), [3, 32]);
        assert_eq!(gathered.dtype, DType::BF16);
        assert!(
            IndexRows
                .gather(&f, source, view(DType::I32, &[1, 3]))
                .is_err()
        );
    }
}

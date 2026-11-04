//! Resampling a learned position table to the grid an image actually has.
//!
//! A vision tower is trained on one grid and serves several, so its learned `[side · side, channels]`
//! position table is resampled bilinearly per image. The table is a device weight and the tower runs
//! inside a captured forward, so the arithmetic cannot go through the host: the caller builds the
//! four tap indices and four `f32` weights per output point once per grid (they depend on the grid,
//! not on the image) and this layer launches `ops::position::PositionInterpolate` over them.
//!
//! The layer carries no weights and no state beyond the compiled form: which dtype, which feature
//! width, and the two grids. The target grid is rectangular — a tower serves `24×40` as readily as a
//! square one — while the learned table is square, because the checkpoint's is.

use ops::DType;
use ops::TensorView;
use ops::position::{Descriptor, PositionInterpolate as PositionInterpolateOp};

use crate::error::{Error, Result};
use crate::forward::Forward;

/// Elements one thread block handles.
const BLOCK: u32 = 1024;

/// Bilinear resample of a learned position table, one launch per grid.
#[derive(Clone, Copy)]
pub struct PositionInterpolate {
    datatype: DType,
    channels: u32,
    in_side: u32,
    out_rows: u32,
    out_columns: u32,
}

impl PositionInterpolate {
    /// A resample from an `in_side`² grid to an `out_rows` × `out_columns` grid of rows `channels`
    /// wide.
    pub fn new(
        dtype: DType,
        channels: u64,
        in_side: u32,
        out_rows: u32,
        out_columns: u32,
    ) -> Result<Self> {
        if in_side == 0 || out_rows == 0 || out_columns == 0 {
            return Err(Error::shape(
                "PositionInterpolate",
                "a grid with no points has no table to resample".to_owned(),
            ));
        }
        let channels = u32::try_from(channels).map_err(|_| {
            Error::shape(
                "PositionInterpolate",
                format!("a table {channels} elements wide is wider than a descriptor holds"),
            )
        })?;
        Ok(Self {
            datatype: dtype,
            channels,
            in_side,
            out_rows,
            out_columns,
        })
    }

    /// `out[j] = Σ_{t<4} weights[j, t] · table[indices[j, t]]`.
    ///
    /// `table` is `[in_side², channels]`, `indices` and `weights` are
    /// `[out_rows · out_columns, 4]` (i32 and f32), and `out` is `[out_rows · out_columns, channels]`.
    /// The index and weight tables are the caller's: they belong to the grid, and one pair serves
    /// every image of that grid.
    pub fn forward(
        &self,
        f: &Forward,
        table: TensorView,
        indices: TensorView,
        weights: TensorView,
        out: TensorView,
    ) -> Result<()> {
        let descriptor = Descriptor {
            dtype: self.datatype,
            channels: self.channels,
            in_side: self.in_side,
            out_rows: self.out_rows,
            out_columns: self.out_columns,
            block: BLOCK,
            pdl: false,
        };
        f.operate::<PositionInterpolateOp>(
            &descriptor,
            ops::position::Args {
                table,
                indices,
                weights,
                out,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_grid_or_an_oversized_width_is_refused() {
        assert!(PositionInterpolate::new(DType::BF16, 1280, 24, 24, 40).is_ok());
        assert!(PositionInterpolate::new(DType::BF16, 1280, 0, 24, 40).is_err());
        assert!(PositionInterpolate::new(DType::BF16, 1280, 24, 0, 40).is_err());
        assert!(PositionInterpolate::new(DType::BF16, 1280, 24, 24, 0).is_err());
        assert!(
            PositionInterpolate::new(DType::BF16, u64::from(u32::MAX) + 1, 24, 24, 40).is_err()
        );
    }
}

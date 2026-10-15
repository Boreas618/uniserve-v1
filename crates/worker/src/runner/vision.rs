//! Vision feature blocks in a language sequence.
//!
//! A context prefill places each input image's encoder features as one
//! block of its sequence, and a feedback prefill places a generated image's
//! features after the image. The model's [`FeatureInjection`] policy fixes
//! the block's tokens and positions:
//!
//! - its layout: `Direct` places the features alone; `Framed` puts a start
//!   marker before them and an end marker after. A block that closes a
//!   generated image always ends with the end marker.
//! - its positions: `Temporal` gives every slot the conditioning position;
//!   `Sequential` consecutive positions from it, as text tokens take;
//!   `TemporalSpatial` three axes (temporal, row, column), the feature slots
//!   taking their raster coordinates in the feature grid and markers zero
//!   spatial coordinates. A closing block's slots sit one temporal position
//!   further, and its end marker two past the conditioning position.
//!
//! Feature slots carry placeholder token id 1; the text runner replaces
//! their embeddings with the features (the block's mask).

use std::ops::Range;

use crate::{Error, Result};

/// How features enter the sequence (`uniserve.processing.FeatureLayout`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureLayout {
    Direct,
    Framed,
}

/// The position coordinates of inserted features (`PositionLayout`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PositionLayout {
    Temporal,
    TemporalSpatial,
    Sequential,
}

/// A model's feature injection policy, with its marker tokens resolved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeatureInjection {
    pub layout: FeatureLayout,
    pub positions: PositionLayout,
    pub start_token: Option<u32>,
    pub end_token: Option<u32>,
}

/// The positions of a block's slots: one axis, or the three rotary axes
/// (temporal, row, column).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlockPositions {
    Linear(Vec<i64>),
    Spatial([Vec<i64>; 3]),
}

/// One vision block's slots.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisionBlock {
    /// Token ids: markers, and placeholder 1 at feature slots.
    pub token_ids: Vec<i64>,
    /// The feature slots, whose embeddings the features replace.
    pub features: Range<usize>,
    pub positions: BlockPositions,
}

impl VisionBlock {
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }
}

impl FeatureInjection {
    /// Lay out a block of `feature_tokens` features at
    /// `conditioning_position`.
    ///
    /// `grid` is the image's raw patch grid (rows, columns) before the
    /// encoder's pooling, required by `TemporalSpatial` positions: the
    /// feature grid is that grid divided by the per-axis factor that maps
    /// its patch count onto the feature count.
    ///
    /// # Errors
    /// `Invalid` when the block has no features, a required marker token is
    /// unresolved, or the feature count does not divide the patch grid into
    /// a square pooling.
    pub fn block(
        &self,
        feature_tokens: usize,
        conditioning_position: u64,
        close_image: bool,
        grid: Option<[u32; 2]>,
    ) -> Result<VisionBlock> {
        if feature_tokens == 0 {
            return Err(Error::Invalid(
                "vision features must have shape [tokens, hidden]".into(),
            ));
        }
        let leading = self.layout == FeatureLayout::Framed;
        let trailing = leading || close_image;
        let query = usize::from(leading) + feature_tokens + usize::from(trailing);
        let marker = |token: Option<u32>| {
            token.map(i64::from).ok_or_else(|| {
                Error::Invalid("feature injection requires a resolved marker token id".into())
            })
        };
        let mut token_ids = vec![1; query];
        if leading {
            token_ids[0] = marker(self.start_token)?;
        }
        if trailing {
            token_ids[query - 1] = marker(self.end_token)?;
        }
        let begin = usize::from(leading);
        let start = conditioning_position as i64;
        let positions = match self.positions {
            PositionLayout::Temporal => BlockPositions::Linear(vec![start; query]),
            PositionLayout::Sequential => {
                BlockPositions::Linear((start..start + query as i64).collect())
            }
            PositionLayout::TemporalSpatial => {
                let [_, columns] = feature_grid(grid, feature_tokens)?;
                let mut temporal = vec![start + i64::from(close_image); query];
                let mut row = vec![0; query];
                let mut column = vec![0; query];
                for index in 0..feature_tokens {
                    row[begin + index] = (index / columns) as i64;
                    column[begin + index] = (index % columns) as i64;
                }
                if trailing && close_image {
                    temporal[query - 1] = start + 2;
                }
                BlockPositions::Spatial([temporal, row, column])
            }
        };
        Ok(VisionBlock {
            token_ids,
            features: begin..begin + feature_tokens,
            positions,
        })
    }
}

/// The feature grid of a raw patch grid pooled into `feature_tokens`
/// features: each axis divided by the square root of the pooling.
fn feature_grid(grid: Option<[u32; 2]>, feature_tokens: usize) -> Result<[usize; 2]> {
    let [raw_rows, raw_columns] = grid.ok_or_else(|| {
        Error::Invalid("temporal-spatial feature injection requires a patch image transform".into())
    })?;
    let (raw_rows, raw_columns) = (raw_rows as usize, raw_columns as usize);
    let patches = raw_rows * raw_columns;
    let (pooling, remainder) = (patches / feature_tokens, patches % feature_tokens);
    let factor = pooling.isqrt();
    if remainder != 0 || factor == 0 || factor * factor != pooling {
        return Err(Error::Invalid(
            "vision feature count does not align with its patch grid".into(),
        ));
    }
    let (rows, columns) = (raw_rows / factor, raw_columns / factor);
    if rows * columns != feature_tokens {
        return Err(Error::Invalid("vision output grid is not integral".into()));
    }
    Ok([rows, columns])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn injection(layout: FeatureLayout, positions: PositionLayout) -> FeatureInjection {
        FeatureInjection {
            layout,
            positions,
            start_token: Some(151_652),
            end_token: Some(151_653),
        }
    }

    #[test]
    fn framed_temporal_blocks_hold_their_features_between_markers() -> Result<()> {
        let block =
            injection(FeatureLayout::Framed, PositionLayout::Temporal).block(3, 9, false, None)?;
        assert_eq!(block.token_ids, vec![151_652, 1, 1, 1, 151_653]);
        assert_eq!(block.features, 1..4);
        assert_eq!(block.positions, BlockPositions::Linear(vec![9; 5]));
        Ok(())
    }

    #[test]
    fn sequential_blocks_take_consecutive_positions() -> Result<()> {
        let block = injection(FeatureLayout::Direct, PositionLayout::Sequential)
            .block(4, 10, false, None)?;
        assert_eq!(block.token_ids, vec![1; 4]);
        assert_eq!(
            block.positions,
            BlockPositions::Linear(vec![10, 11, 12, 13])
        );
        Ok(())
    }

    #[test]
    fn spatial_blocks_take_raster_coordinates_of_the_pooled_grid() -> Result<()> {
        // A 4x6 patch grid pooled 2x2 into 2x3 features.
        let block = injection(FeatureLayout::Direct, PositionLayout::TemporalSpatial).block(
            6,
            5,
            false,
            Some([4, 6]),
        )?;
        assert_eq!(block.features, 0..6);
        assert_eq!(
            block.positions,
            BlockPositions::Spatial([vec![5; 6], vec![0, 0, 0, 1, 1, 1], vec![0, 1, 2, 0, 1, 2]])
        );

        // A closing block sits one temporal position later and ends with
        // the end marker two past the conditioning position.
        let block = injection(FeatureLayout::Direct, PositionLayout::TemporalSpatial).block(
            6,
            5,
            true,
            Some([4, 6]),
        )?;
        assert_eq!(block.token_ids, vec![1, 1, 1, 1, 1, 1, 151_653]);
        assert_eq!(
            block.positions,
            BlockPositions::Spatial([
                vec![6, 6, 6, 6, 6, 6, 7],
                vec![0, 0, 0, 1, 1, 1, 0],
                vec![0, 1, 2, 0, 1, 2, 0],
            ])
        );
        Ok(())
    }

    #[test]
    fn grids_that_do_not_pool_into_the_features_are_refused() {
        let spatial = injection(FeatureLayout::Direct, PositionLayout::TemporalSpatial);
        // 24 patches over 4 features is a pooling of 6, not a square.
        assert!(spatial.block(4, 0, false, Some([4, 6])).is_err());
        assert!(spatial.block(6, 0, false, None).is_err());
        assert!(spatial.block(0, 0, false, Some([4, 6])).is_err());
        let unresolved = FeatureInjection {
            end_token: None,
            ..injection(FeatureLayout::Framed, PositionLayout::Temporal)
        };
        assert!(unresolved.block(2, 0, false, None).is_err());
    }
}

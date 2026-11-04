//! Selecting the key tiles a video sparse attention reads (VSA).
//!
//! The sequence is cut into `tile_rows`-row tiles and a query tile attends only to the key tiles its
//! pooled scores select. Five programs do it, none of them allocating, all of them leaving their
//! indices on the device so a captured graph replays the selection (specs/worker.md 9, "VSA 分块";
//! `ops::segments`):
//!
//! ```text
//! prepare     normalize and rotate Q and K in place, zero each tile's padding rows, pool the means
//! zero        zero each tile's padding rows, in place
//! add_gated   fold the compressed branch's gated rows into a tile's rows, in place
//! compose     write one row interval of every member's gated rows, member-major
//! select      choose each query tile's key tiles from its pooled scores
//! ```
//!
//! The layer holds the geometry — tile rows, heads, features, the rotation's extent — and the
//! epsilon of the Q/K normalization. Every activation, every score and every derived table is the
//! caller's view: the block composes this layer with its attention projections and its cache, and
//! the executor owns the buffers.

use ops::TensorView;
use ops::descriptor::Float;
use ops::segments::{self, MAX_MEMBERS};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::weights::Weight;

/// The rows one tile holds: the block size VSA selects with.
const DEFAULT_TILE: u32 = 64;

/// The tile-selection layer of a video sparse attention block.
pub struct VsaTiles {
    tile_rows: u32,
    heads: u32,
    width: u32,
    eps: f32,
    rotated: u32,
    norm: bool,
    block_rows: u32,
}

impl VsaTiles {
    /// The selection of `heads` heads of `width` features over `tile_rows`-row tiles.
    pub fn new(tile_rows: u32, heads: u32, width: u32, eps: f32) -> Result<Self> {
        if tile_rows == 0 || heads == 0 || width == 0 {
            return Err(Error::shape(
                "VsaTiles",
                "a tile, a head count and a feature count are each at least one".to_owned(),
            ));
        }
        if !tile_rows.is_multiple_of(segments::BLOCK_ROWS) {
            return Err(Error::shape(
                "VsaTiles",
                format!(
                    "a tile of {tile_rows} rows is not a whole number of {} programmes",
                    segments::BLOCK_ROWS
                ),
            ));
        }
        Ok(Self {
            tile_rows,
            heads,
            width,
            eps,
            rotated: 0,
            norm: false,
            block_rows: segments::BLOCK_ROWS,
        })
    }

    /// The default tile of a video sparse attention block.
    pub fn with_default_tile(heads: u32, width: u32, eps: f32) -> Result<Self> {
        Self::new(DEFAULT_TILE, heads, width, eps)
    }

    /// Normalize Q and K with their learned weights before pooling.
    pub fn with_norm(mut self) -> Self {
        self.norm = true;
        self
    }

    /// Rotate the first `rotated` features of Q and K with the factor tables; implies the norm, as
    /// the reference applies them together.
    pub fn with_rotation(mut self, rotated: u32) -> Result<Self> {
        if rotated == 0 || rotated > self.width || !rotated.is_multiple_of(2) {
            return Err(Error::shape(
                "VsaTiles",
                format!("a rotation of {rotated} features of {}", self.width),
            ));
        }
        self.rotated = rotated;
        self.norm = true;
        Ok(self)
    }

    /// Rows of one tile.
    pub const fn tile_rows(&self) -> u32 {
        self.tile_rows
    }

    /// Heads per row.
    pub const fn heads(&self) -> u32 {
        self.heads
    }

    /// Features per head.
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// What a preparation needs of its at-rest arguments, checked before any launch.
    ///
    /// Split from [`Self::prepare`] because it needs no device: a layer's argument contract is
    /// assertable on the host, and a caller that gets it wrong is told before a kernel is resolved.
    pub fn check_prepare(
        &self,
        query_weight: bool,
        key_weight: bool,
        cosine: Option<TensorView>,
        sine: Option<TensorView>,
    ) -> Result<()> {
        if self.norm != (query_weight && key_weight) {
            return Err(Error::shape(
                "VsaTiles",
                "the Q/K normalization weights are present exactly when the layer normalizes"
                    .to_owned(),
            ));
        }
        if self.rotated != 0 && (cosine.is_none() || sine.is_none()) {
            return Err(Error::shape(
                "VsaTiles",
                "a rotated preparation needs both factor tables".to_owned(),
            ));
        }
        Ok(())
    }

    /// Normalize, rotate, zero and pool whole tiles of projected Q/K/V rows.
    ///
    /// `query`, `key` and `value` are `[tiles · tile_rows, heads, width]` and are modified in place;
    /// `sizes` is `[tiles]` i32; the three pooled views are `[tiles, heads, width]` FP32. The
    /// normalization weights and the factor tables are present exactly when the layer normalizes and
    /// rotates.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &self,
        f: &Forward,
        query: TensorView,
        key: TensorView,
        value: TensorView,
        query_weight: Option<&Weight>,
        key_weight: Option<&Weight>,
        cosine: Option<TensorView>,
        sine: Option<TensorView>,
        sizes: TensorView,
        pooled_query: TensorView,
        pooled_key: TensorView,
        pooled_value: TensorView,
    ) -> Result<()> {
        self.check_prepare(query_weight.is_some(), key_weight.is_some(), cosine, sine)?;
        let descriptor = segments::PrepareTilesDescriptor {
            dtype: query.dtype,
            heads: self.heads,
            width: self.width,
            rotated: self.rotated,
            tile_rows: self.tile_rows,
            block_rows: self.block_rows,
            epsilon: Float::from(self.eps),
            norm: self.norm,
            query_row_stride: query.strides.get(0),
            query_head_stride: query.strides.get(1),
            key_row_stride: key.strides.get(0),
            key_head_stride: key.strides.get(1),
            value_row_stride: value.strides.get(0),
            value_head_stride: value.strides.get(1),
            factor_stride: cosine.map(|table| table.strides.get(0)).unwrap_or(0),
        };
        let args = segments::PrepareTilesArgs {
            query,
            key,
            value,
            query_weight: query_weight.map(Weight::view),
            key_weight: key_weight.map(Weight::view),
            cosine,
            sine,
            sizes,
            pooled_query,
            pooled_key,
            pooled_value,
        };
        f.operate::<segments::VsaPrepareTiles>(&descriptor, args)
    }

    /// Zero the rows of every tile past its valid size, in place.
    pub fn zero_padding(&self, f: &Forward, values: TensorView, sizes: TensorView) -> Result<()> {
        let descriptor = segments::ZeroPaddingDescriptor {
            dtype: values.dtype,
            heads: self.heads,
            tile_rows: self.tile_rows,
            width: self.width,
            block_rows: self.block_rows,
        };
        let args = segments::ZeroPaddingArgs { values, sizes };
        f.operate::<segments::VsaZeroPadding>(&descriptor, args)
    }

    /// Fold the compressed branch's gated rows into each tile's rows, in place.
    pub fn add_gated(
        &self,
        f: &Forward,
        output: TensorView,
        compressed: TensorView,
        gate: TensorView,
    ) -> Result<()> {
        let descriptor = segments::AddGatedTilesDescriptor {
            dtype: output.dtype,
            tile_rows: self.tile_rows,
            width: self.width,
            block_rows: self.block_rows,
        };
        let args = segments::AddGatedTilesArgs {
            output,
            compressed,
            gate,
        };
        f.operate::<segments::VsaAddGatedTiles>(&descriptor, args)
    }

    /// How many members a composition may name, checked before any launch.
    ///
    /// A tensor-parallel group's membership is the caller's, so the count is the caller's argument
    /// and its refusal belongs where a host can see it.
    pub fn check_members(&self, members: usize) -> Result<()> {
        if members == 0 || members > MAX_MEMBERS {
            return Err(Error::shape(
                "VsaTiles",
                format!("{members} members, one to {MAX_MEMBERS}"),
            ));
        }
        Ok(())
    }

    /// Write one row interval of every member's gated rows, member-major.
    ///
    /// `slots` names the payload slot of each member, in member order; a tensor-parallel group has
    /// one member per rank.
    #[allow(clippy::too_many_arguments)]
    pub fn compose(
        &self,
        f: &Forward,
        output: TensorView,
        fine: TensorView,
        compressed: TensorView,
        gate: TensorView,
        owner_rows: u64,
        start: u64,
        count: u64,
        slots: &[u32],
    ) -> Result<()> {
        self.check_members(slots.len())?;
        let mut table = [0_u32; MAX_MEMBERS];
        table[..slots.len()].copy_from_slice(slots);
        let descriptor = segments::ComposeGatedRowsDescriptor {
            dtype: output.dtype,
            heads: self.heads,
            tile_rows: self.tile_rows,
            width: self.width,
            block_rows: self.block_rows,
            slots: table,
            members: slots.len() as u32,
        };
        let args = segments::ComposeGatedRowsArgs {
            output,
            fine,
            compressed,
            gate,
            owner_rows,
            start,
            count,
        };
        f.operate::<segments::VsaComposeGatedRows>(&descriptor, args)
    }

    /// Contract the pooled query and key tiles into the scores `select` ranks.
    ///
    /// `pooled_query` is `[query tiles, heads, width]` and `pooled_key` `[tiles, heads, width]`, both
    /// FP32 — the layout and the type [`Self::prepare`] writes; `scores` is
    /// `[heads, query tiles, tiles]` FP32, contiguous, written here and read by [`Self::select`]. One
    /// dot product per (query tile, key tile) and head — no scale and no softmax, because the
    /// selection only compares and the attention scales its own products.
    pub fn scores(
        &self,
        f: &Forward,
        pooled_query: TensorView,
        pooled_key: TensorView,
        scores: TensorView,
    ) -> Result<()> {
        if pooled_query.shape.rank() != 3 {
            return Err(Error::shape(
                "VsaTiles",
                format!(
                    "the pooled query is [query tiles, heads, width], not {:?}",
                    pooled_query.shape.dims()
                ),
            ));
        }
        let width = u32::try_from(pooled_query.shape.dim(2)).map_err(|_| {
            Error::shape(
                "VsaTiles",
                format!("{} features per head", pooled_query.shape.dim(2)),
            )
        })?;
        let descriptor =
            segments::ScoresDescriptor::new(self.heads, width, width.next_power_of_two(), false);
        let args = segments::ScoresArgs {
            pooled_query,
            pooled_key,
            out: scores,
        };
        f.operate::<segments::VsaScores>(&descriptor, args)
    }

    /// Choose each query tile's key tiles from its pooled scores.
    ///
    /// `scores` is `[heads, query tiles, tiles]` FP32 contiguous; `segments` and `sizes` are
    /// `[tiles]` i32; `keep` is `[segments]` i32. The layer writes `indices`
    /// `[heads, query tiles, tiles]` i32 (kept tiles first, in ascending order) and `counts`
    /// `[heads, query tiles]` i32. `query_offset` is the sequence tile of the first query tile.
    #[allow(clippy::too_many_arguments)]
    pub fn select(
        &self,
        f: &Forward,
        scores: TensorView,
        segments: TensorView,
        sizes: TensorView,
        keep: TensorView,
        indices: TensorView,
        counts: TensorView,
        query_offset: u64,
    ) -> Result<()> {
        if scores.shape.rank() != 3 {
            return Err(Error::shape(
                "VsaTiles",
                format!(
                    "the pooled scores are [heads, query tiles, tiles], not {:?}",
                    scores.shape.dims()
                ),
            ));
        }
        let tiles = scores.shape.dim(2);
        let descriptor = segments::SelectTilesDescriptor {
            block: u32::try_from(tiles.max(1).next_power_of_two()).map_err(|_| {
                Error::shape("VsaTiles", format!("{tiles} key tiles in one query tile"))
            })?,
        };
        let args = segments::SelectTilesArgs {
            scores,
            segments,
            sizes,
            keep,
            indices,
            counts,
            query_offset,
        };
        f.operate::<segments::VsaSelectTiles>(&descriptor, args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ops::{DType, DevicePtr, Shape};

    /// A view of `shape` at a host address: these tests never launch, so the address is a marker.
    fn view(shape: &[u64]) -> TensorView {
        TensorView::new(DevicePtr::new(0x1000), DType::F32, Shape::new(shape))
    }

    #[test]
    fn a_tile_is_whole_programme_rows_and_positive_everywhere() {
        let default = VsaTiles::with_default_tile(8, 256, 1e-6).unwrap();
        assert_eq!(default.tile_rows(), DEFAULT_TILE);
        assert_eq!(default.heads(), 8);
        assert_eq!(default.width(), 256);
        // A tile of 64 rows and a 128-row checkpoint both serve: tile rows are the checkpoint's.
        assert!(VsaTiles::new(128, 8, 256, 1e-6).is_ok());
        assert!(VsaTiles::new(0, 8, 256, 1e-6).is_err());
        assert!(VsaTiles::new(64, 0, 256, 1e-6).is_err());
        assert!(VsaTiles::new(64, 8, 0, 1e-6).is_err());
        let text = match VsaTiles::new(96, 8, 256, 1e-6) {
            Ok(_) => panic!("96 is not a whole number of programme rows"),
            Err(error) => format!("{error}"),
        };
        assert!(text.contains("96"), "{text}");
    }

    /// A rotation reads the first features of a head in pairs, and implies the normalization the
    /// reference applies with it: a layer that rotates without normalizing is not a case.
    #[test]
    fn a_rotation_is_even_bounded_by_the_width_and_implies_the_norm() {
        let rotated = VsaTiles::new(64, 8, 256, 1e-6)
            .unwrap()
            .with_rotation(96)
            .unwrap();
        assert_eq!(rotated.rotated, 96);
        assert!(rotated.norm, "a rotation implies the norm");
        assert!(
            VsaTiles::new(64, 8, 256, 1e-6)
                .unwrap()
                .with_rotation(0)
                .is_err()
        );
        assert!(
            VsaTiles::new(64, 8, 256, 1e-6)
                .unwrap()
                .with_rotation(97)
                .is_err(),
            "a rotation reads pairs of features"
        );
        assert!(
            VsaTiles::new(64, 8, 256, 1e-6)
                .unwrap()
                .with_rotation(512)
                .is_err(),
            "a rotation of more features than a head has"
        );
    }

    /// The weights are present exactly when the layer normalizes, and a rotation needs both factor
    /// tables: a preparation missing either would read a null view, so both are refused by name.
    #[test]
    fn a_preparation_names_its_weights_and_tables_exactly() {
        let plain = VsaTiles::new(64, 8, 256, 1e-6).unwrap();
        assert!(plain.check_prepare(false, false, None, None).is_ok());
        assert!(plain.check_prepare(true, true, None, None).is_err());
        assert!(plain.check_prepare(true, false, None, None).is_err());

        let normed = VsaTiles::new(64, 8, 256, 1e-6).unwrap().with_norm();
        assert!(normed.check_prepare(true, true, None, None).is_ok());
        assert!(normed.check_prepare(false, false, None, None).is_err());
        assert!(
            normed.check_prepare(true, false, None, None).is_err(),
            "one weight is not both halves of the normalization"
        );

        let rotated = VsaTiles::new(64, 8, 256, 1e-6)
            .unwrap()
            .with_rotation(96)
            .unwrap();
        let (cosine, sine) = (view(&[4096, 96]), view(&[4096, 96]));
        assert!(
            rotated
                .check_prepare(true, true, Some(cosine), Some(sine))
                .is_ok()
        );
        let missing = rotated.check_prepare(true, true, Some(cosine), None);
        assert!(
            format!("{}", missing.unwrap_err()).contains("factor tables"),
            "both tables are the rotation's"
        );
    }

    /// A composition names one member per rank of a tensor-parallel group: at least one, at most the
    /// operator's table.
    #[test]
    fn a_composition_names_one_member_per_rank_and_no_more_than_the_table() {
        let tiles = VsaTiles::new(64, 8, 256, 1e-6).unwrap();
        assert!(tiles.check_members(1).is_ok());
        assert!(tiles.check_members(MAX_MEMBERS).is_ok());
        let none = tiles.check_members(0);
        assert!(
            format!("{}", none.unwrap_err()).contains("one to"),
            "no member is not a composition"
        );
        let too_many = tiles.check_members(MAX_MEMBERS + 1);
        assert!(
            format!("{}", too_many.unwrap_err()).contains(&MAX_MEMBERS.to_string()),
            "the table's size is named"
        );
    }
}

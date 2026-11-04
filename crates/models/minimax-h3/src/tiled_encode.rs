//! The video encoder's tiled path: a raster larger than one tile becomes its latent by tiles that
//! cross-fade where they meet.
//!
//! [`crate::encoding::VideoEncoder::encode`] encodes **one** tile; this module is the caller it
//! describes. The reference
//! (`uniserve/nn/vae/tiled.py:168-303`, `TiledEncoder.encode`) splits an aligned raster into
//! overlapping tiles (`split_tiles`, `:9-44`), encodes the tile rows a band of latent rows depends
//! on, blends the tiles vertically and then horizontally (`stitch_tiles`, `:46-81`), and crops the
//! band. The geometry of that split is [`crate::encoding::Grid`], which the video encoder already
//! owns because [`crate::video_vae::Seams`] is the decoder's split too.
//!
//! What is here is the plan a driver executes: for each tile, where its latents land in the raster,
//! how far it fades against the tile above and the tile to its left, and which tiles a band of
//! latent rows reads. It holds no buffers and launches nothing, so it can be checked against the
//! reference's own numbers on the host.
//!
//! Two properties the reference states and this plan carries:
//!
//! - **A band equals the whole.** Encoding latent rows `[start, stop)` on its own encodes only the
//!   tile rows those rows read and crops them from the same arithmetic, so the rows are identical to
//!   the whole encoding's (`:236-240`). [`ClipEncoder::band`] states the tile rows and the crop.
//! - **Neighbours stay as decoded.** A tile's vertical fade reads the tile above *as decoded*, and
//!   its horizontal fade reads the tile to its left *as decoded*, not as already faded, or the
//!   overlap where two seams cross is computed in a different order from the seams themselves
//!   (`:51-56`). [`Tile::above`] and [`Tile::left`] name those reads, and the driver keeps a second
//!   raster for them.

use crate::encoding::{Grid, SPATIAL_RATIO, VideoEncoder};
use crate::{Error, Result};
use nn::layers::CrossFade;
use nn::{DType, Forward, Shape, Strides, TensorView};

/// One tile of a tiled encoding: the latent block it writes and the seams it fades over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile {
    /// The tile row, counting from the raster's first tile row.
    pub row: usize,
    /// The tile column, counting from the raster's first tile column.
    pub column: usize,
    /// The latent rows the tile occupies, as `(start, length)`, in raster coordinates.
    pub rows: (u32, u32),
    /// The latent columns the tile occupies, as `(start, length)`.
    pub columns: (u32, u32),
    /// The latent rows it cross-fades with the tile above, at its top: `overlap / ratio` of them.
    ///
    /// Zero on the raster's first tile row, and on the first tile row of a band that begins outside
    /// a seam, which has nothing above it to fade against.
    pub above: u32,
    /// The latent columns it cross-fades with the tile to its left, at its leading edge.
    ///
    /// Zero on the raster's first tile column.
    pub left: u32,
}

impl Tile {
    /// Whether the tile has a neighbour above it in the run, and so a fade to make.
    pub const fn fades_above(&self) -> bool {
        self.above != 0
    }

    /// Whether the tile has a neighbour to its left in the run, and so a fade to make.
    pub const fn fades_left(&self) -> bool {
        self.left != 0
    }
}

/// The tiles of one tiled encoding, and the crop that turns a run into the rows asked for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    /// Every tile of the run, in the order the reference encodes and stitches them: row-major.
    pub tiles: Vec<Tile>,
    /// The latent row of the run's first tile row, which the crop subtracts.
    pub origin: u32,
    /// The latent rows of the run the caller keeps, `[start, stop)`, in raster coordinates.
    pub crop: (u32, u32),
}

impl Plan {
    /// The latent rows the run covers before the crop, as `(origin, end)`, `end` exclusive.
    ///
    /// A run of `end - origin` latent rows is what the caller's two rasters must hold; the crop then
    /// keeps `crop.start - origin` to `crop.stop - origin`. The pair is stated as an origin and an
    /// end rather than a start and a length because the tiles' rows are in raster coordinates and a
    /// length invites the off-by-one.
    pub fn extent(&self) -> (u32, u32) {
        let end = self
            .tiles
            .iter()
            .map(|tile| tile.rows.0 + tile.rows.1)
            .max()
            .unwrap_or(self.origin);
        (self.origin, end)
    }
}

/// The device views one tiled encoding reads and writes, as the caller uploaded and reserved them.
///
/// The raster is of the **run** - `Plan::extent` rows - and FP32, the type the video encoder computes
/// in. The caller keeps the rows `Plan::crop` names, shifted back by `Plan::origin`.
#[derive(Clone, Copy, Debug)]
pub struct Scene<'a> {
    /// Tile `i`'s pixels, `[1, in_channels, frames, tile_height, tile_width]`, in plan order.
    pub pixels: &'a [TensorView],
    /// The run's stitched latents, `[channels, frames, run_rows, latent_columns]`.
    pub blended: TensorView,
}

/// The tiled encoding of one raster: the tile grid, and the band arithmetic over it.
///
/// The raster's extents are input pixels and must be multiples of the spatial compression, which is
/// what [`Grid::of`] checks; the latents are the raster divided by it.
#[derive(Clone)]
pub struct ClipEncoder {
    grid: Grid,
    latent: (u32, u32),
    /// One compiled blend per seam width and axis, as `(extent, axis, layer)`.
    blends: Vec<(u32, u32, CrossFade)>,
}

impl ClipEncoder {
    /// The tiled encoding of a `height` by `width` raster, in input pixels.
    pub fn new(height: u32, width: u32) -> Result<Self> {
        let grid = Grid::of(height, width)?;
        let encoder = Self {
            grid,
            latent: (height / SPATIAL_RATIO, width / SPATIAL_RATIO),
            blends: Vec::new(),
        };
        let blends = encoder
            .blend_extents()
            .into_iter()
            .map(|(extent, axis)| {
                CrossFade::new(DType::F32, extent, axis)
                    .map(|layer| (extent, axis, layer))
                    .map_err(Error::from)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { blends, ..encoder })
    }

    /// The latent raster one whole encoding produces, `(rows, columns)`.
    pub const fn latent_extent(&self) -> (u32, u32) {
        self.latent
    }

    /// The raster's tile grid.
    pub const fn grid(&self) -> &Grid {
        &self.grid
    }

    /// The seams a run of this raster blends over, as `(extent, axis)` pairs.
    ///
    /// One compiled blend serves every seam of the same width, so a driver compiles these and no
    /// more: the rows axis is `2` and the columns axis `3`, as [`nn::layers::CrossFade`] numbers
    /// them, and a raster inside one tile has none.
    pub fn blend_extents(&self) -> Vec<(u32, u32)> {
        let mut extents = Vec::new();
        for (axis, seams) in [(2, &self.grid.rows), (3, &self.grid.columns)] {
            for extent in Grid::blends(seams) {
                if extent != 0 && !extents.contains(&(extent, axis)) {
                    extents.push((extent, axis));
                }
            }
        }
        extents.sort_unstable();
        extents
    }

    /// Encode one run of the raster: every tile of `plan`, stitched into `scene.blended`.
    ///
    /// The tiles are written in [`Plan::tiles`] order, because a seam blend reads the block above it
    /// and the block to its left, and both must be read **as decoded** rather than as already faded:
    /// that is what makes the overlap where two seams cross follow the same arithmetic as the seams
    /// themselves (`uniserve/nn/vae/tiled.py:51-56`). `nn::layers::CrossFade` reads its two sources as
    /// contiguous blocks and writes its destination through the destination's strides, so the driver
    /// keeps the decoded tiles in a ring of contiguous buffers and blends from there into the raster.
    ///
    /// Two buffers per tile column, one row of a two-row ring: row `r` reads row `r - 1`'s blocks and
    /// overwrites row `r - 2`'s, which nothing reads again. One more block holds a tile whose fade is
    /// both ways, because the second fade reads the first's result. A run's first tile row has nothing
    /// above it, so its tiles blend only to the left; a run's first tile has nothing at all to blend
    /// against, and its block reaches the raster through a blend of `extent` 1. That position takes
    /// weight one on the **previous** source and zero on the current one
    /// (`cross_fade.py`: `previous[along - extent + i] · (1 - i/extent) + current[i] · (i/extent)`),
    /// so the pass-through is exact because the same block is passed as both sources - not because a
    /// blend of one position ignores one of them.
    pub fn encode(
        &self,
        f: &Forward,
        encoder: &VideoEncoder,
        plan: &Plan,
        scene: Scene<'_>,
    ) -> Result<()> {
        let Scene { pixels, blended } = scene;
        if blended.shape.rank() != 4 {
            return Err(Error::config(format!(
                "the raster is {:?}, not [channels, frames, rows, columns]",
                blended.shape.dims()
            )));
        }
        if blended.dtype != DType::F32 {
            return Err(Error::config(format!(
                "the raster is {}, not the encoder's F32",
                blended.dtype
            )));
        }
        if pixels.len() != plan.tiles.len() {
            return Err(Error::config(format!(
                "{} pixel views for {} tiles",
                pixels.len(),
                plan.tiles.len()
            )));
        }
        let (origin, end) = plan.extent();
        let rows = u64::from(end - origin);
        let columns = u64::from(self.latent.1);
        if blended.shape.dim(2) < rows || blended.shape.dim(3) < columns {
            return Err(Error::config(format!(
                "a run of {rows} by {columns} latent rows does not fit {:?}",
                blended.shape.dims()
            )));
        }
        let tile_rows = plan.tiles.first().map(|tile| tile.rows.1).unwrap_or(0);
        let tile_columns = plan.tiles.first().map(|tile| tile.columns.1).unwrap_or(0);
        if tile_rows == 0 || tile_columns == 0 {
            return Err(Error::config("a tiled encoding without a tile".to_owned()));
        }

        let block = Shape::new(&[
            blended.shape.dim(0),
            blended.shape.dim(1),
            u64::from(tile_rows),
            u64::from(tile_columns),
        ]);
        // The two decoded rows of the ring, and the block a two-way fade hands to its second blend.
        let mut ring = Vec::with_capacity(2);
        for _ in 0..2 {
            let mut row = Vec::with_capacity(self.grid.columns.len());
            for _ in 0..self.grid.columns.len() {
                row.push(f.arena().tensor(DType::F32, block)?);
            }
            ring.push(row);
        }
        let handed = f.arena().tensor(DType::F32, block)?;

        // A blend of one position, given the same block twice: weight one on the previous source and
        // zero on the current one, so `decoded · 1 + decoded · 0` places it unchanged. Passing two
        // different blocks here would place the previous one, which is why the call below repeats
        // its view.
        let place = CrossFade::new(DType::F32, 1, 2).map_err(Error::from)?;

        for (index, tile) in plan.tiles.iter().enumerate() {
            if tile.rows.1 != tile_rows || tile.columns.1 != tile_columns {
                return Err(Error::config(format!(
                    "tile {index} is {} by {} latents, not the run's {tile_rows} by {tile_columns}",
                    tile.rows.1, tile.columns.1
                )));
            }
            let decoded = ring[tile.row % 2][tile.column];
            // The encoder reads `[1, channels, frames, rows, columns]`; the block is that, batched by
            // one.
            let strides = decoded.strides.as_slice();
            let out = TensorView::with_strides(
                decoded.element(&[0, 0, 0, 0]),
                DType::F32,
                Shape::new(&[
                    1,
                    block.dim(0),
                    block.dim(1),
                    u64::from(tile_rows),
                    u64::from(tile_columns),
                ]),
                Strides::new(&[strides[0], strides[0], strides[1], strides[2], strides[3]]),
            );
            encoder.encode(f, pixels[index], out)?;

            let placed = self.place_in(blended, plan, tile, block)?;
            let above = match tile.fades_above() {
                true => Some(ring[(tile.row + 1) % 2][tile.column]),
                false => None,
            };
            let left = match tile.fades_left() {
                true => Some(ring[tile.row % 2][tile.column - 1]),
                false => None,
            };
            match (above, left) {
                (Some(above), Some(left)) => {
                    self.blend(tile.above, 2)?
                        .forward(f, Some(above), decoded, handed)?;
                    self.blend(tile.left, 3)?
                        .forward(f, Some(left), handed, placed)?;
                }
                (Some(above), None) => {
                    self.blend(tile.above, 2)?
                        .forward(f, Some(above), decoded, placed)?;
                }
                (None, Some(left)) => {
                    self.blend(tile.left, 3)?
                        .forward(f, Some(left), decoded, placed)?;
                }
                (None, None) => {
                    // The same view twice: at `extent` 1 the previous source carries the whole
                    // weight, so a differing pair would write `decoded`'s neighbour instead.
                    place.forward(f, Some(decoded), decoded, placed)?;
                }
            }
        }
        Ok(())
    }

    /// A view of `raster` holding one tile's block, at the tile's place in the run.
    ///
    /// The view keeps the raster's own strides, so a blend writes through them.
    fn place_in(
        &self,
        raster: TensorView,
        plan: &Plan,
        tile: &Tile,
        block: Shape,
    ) -> Result<TensorView> {
        let rows = u64::from(tile.rows.0) - u64::from(plan.origin);
        let columns = u64::from(tile.columns.0);
        if rows + block.dim(2) > raster.shape.dim(2) || columns + block.dim(3) > raster.shape.dim(3)
        {
            return Err(Error::config(format!(
                "the block at ({rows}, {columns}) of {} by {} does not fit {:?}",
                block.dim(2),
                block.dim(3),
                raster.shape.dims()
            )));
        }
        Ok(TensorView::with_strides(
            raster.element(&[0, 0, rows, columns]),
            raster.dtype,
            Shape::new(&[
                raster.shape.dim(0),
                raster.shape.dim(1),
                block.dim(2),
                block.dim(3),
            ]),
            raster.strides,
        ))
    }

    /// The blend layer of `extent` along `axis`, compiled once with the encoder.
    fn blend(&self, extent: u32, axis: u32) -> Result<CrossFade> {
        self.blends
            .iter()
            .find(|(width, side, _)| *width == extent && *side == axis)
            .map(|(_, _, layer)| *layer)
            .ok_or_else(|| {
                Error::config(format!(
                    "the tiled encoder compiled no {extent}-position blend along axis {axis}"
                ))
            })
    }

    /// Every tile of the whole raster, in row-major order, and the identity crop.
    pub fn plan(&self) -> Plan {
        self.plan_of(
            0,
            self.latent.0,
            (0, self.grid.rows.len().saturating_sub(1)),
        )
    }

    /// The tiles a band of latent rows `[start, stop)` reads, and the crop that keeps those rows.
    ///
    /// The first tile row is the last one starting at or before `start`, preceded by one more row
    /// when `start` falls inside the cross-fade that row shares with it, because those rows read the
    /// row above. The crop is the band shifted by the run's origin, so a caller stitching bands
    /// writes each band's rows where the whole encoding would have them.
    pub fn band(&self, start: u32, stop: u32) -> Result<Plan> {
        if start >= stop || stop > self.latent.0 {
            return Err(Error::config(format!(
                "the latent band [{start}, {stop}) is not a run of the {} rows",
                self.latent.0
            )));
        }
        let (first, last) = self.grid.band(self.latent.0, start, stop)?;
        Ok(self.plan_of(start, stop, (first, last)))
    }

    /// The plan of the tile rows `first..=last`, cropping `[start, stop)`.
    fn plan_of(&self, start: u32, stop: u32, (first, last): (usize, usize)) -> Plan {
        let owned = self.grid.owned(self.latent.0);
        let rows_blend = Grid::blends(&self.grid.rows);
        let columns_blend = Grid::blends(&self.grid.columns);
        let mut tiles = Vec::with_capacity((last + 1 - first) * self.grid.columns.len());
        for row in first..=last {
            for column in 0..self.grid.columns.len() {
                tiles.push(Tile {
                    row,
                    column,
                    // A tile row writes its own latent rows; its lower overlap belongs to the row
                    // below, which fades it, so the run's last row ends at the raster's edge.
                    rows: (owned[row], self.grid.rows.lengths[row] / SPATIAL_RATIO),
                    columns: (
                        self.grid.columns.starts[column] / SPATIAL_RATIO,
                        self.grid.columns.lengths[column] / SPATIAL_RATIO,
                    ),
                    // The fade reads the row above, which only the first row of the run lacks: a
                    // band that starts inside a seam reads the row above it through
                    // `Grid::band`, so that row is in the run.
                    // A tile fades over the seam it shares with the neighbour *before* it: the row
                    // above and the column to its left, each seam its own width
                    // (`uniserve/nn/vae/tiled.py:63-73`).
                    above: if row > first { rows_blend[row - 1] } else { 0 },
                    left: if column > 0 {
                        columns_blend[column - 1]
                    } else {
                        0
                    },
                });
            }
        }
        Plan {
            origin: owned[first],
            crop: (start, stop),
            tiles,
        }
    }
}

#[cfg(test)]
mod tests {
    //! The expectations below are the reference's own numbers, not this module's arithmetic run
    //! backwards: `split_tiles` (`uniserve/nn/vae/tiled.py:9-44`) and the band prelude of
    //! `TiledEncoder.encode` (`:246-262`) were extracted from the reference source and evaluated on
    //! the rasters these tests use - 768 by 1344, 256 by 256, 480 by 832 - before the values were
    //! written down.

    use super::*;

    /// The 768 by 1344 canvas the fixtures use: 48 by 84 latents, four tile rows and seven columns.
    fn canvas() -> ClipEncoder {
        ClipEncoder::new(768, 1344).expect("the verified canvas")
    }

    #[test]
    fn a_raster_within_one_tile_is_one_tile_with_no_seam() {
        let small = ClipEncoder::new(256, 256).expect("one tile");
        let plan = small.plan();
        assert_eq!(plan.tiles.len(), 1);
        let tile = plan.tiles[0];
        assert_eq!(tile.rows, (0, 16));
        assert_eq!(tile.columns, (0, 16));
        assert!(!tile.fades_above() && !tile.fades_left());
        assert_eq!(plan.crop, (0, 16));
    }

    #[test]
    fn a_raster_needs_one_compiled_blend_for_each_seam_width_it_has() {
        // 1344 by 768 has three seams of two widths along the rows and six of two along the columns:
        // four compiled blends serve the whole raster, not ten.
        assert_eq!(
            canvas().blend_extents(),
            vec![(4, 3), (5, 2), (5, 3), (6, 2)]
        );
        // The encoder compiles exactly those blends, once each, so a driver finds every seam it
        // meets and the run pays for four programs rather than one per seam.
        let encoder = canvas();
        let compiled: Vec<(u32, u32)> = encoder
            .blends
            .iter()
            .map(|(extent, axis, _)| (*extent, *axis))
            .collect();
        assert_eq!(compiled, encoder.blend_extents());
        assert_eq!(compiled.len(), 4);
        // A raster inside one tile has no seam at all.
        let small = ClipEncoder::new(256, 256).expect("one tile");
        assert!(small.blend_extents().is_empty());
        assert!(small.blends.is_empty());
    }

    #[test]
    fn a_raster_that_is_not_whole_patches_is_refused_rather_than_truncated() {
        // The compression is 16 pixels; a raster that does not divide by it has no latent extent,
        // and the encoder reads whole patches, so nothing may round the raster down silently.
        assert!(ClipEncoder::new(1000, 700).is_err());
        assert!(ClipEncoder::new(768, 1344).is_ok());
        assert!(ClipEncoder::new(416, 992).is_ok());
    }

    #[test]
    fn the_canvas_plan_covers_every_latent_position() {
        let encoder = canvas();
        assert_eq!(encoder.latent_extent(), (48, 84));
        let plan = encoder.plan();
        assert_eq!(plan.tiles.len(), 4 * 7);
        assert_eq!(plan.extent(), (0, 48));

        // A tile row owns the latent rows up to the next row's start, and the last owns the rest.
        let owned = encoder.grid().owned(48);
        assert_eq!(owned, [0, 10, 21, 32, 48]);
        for tile in &plan.tiles {
            assert_eq!(tile.rows.0, owned[tile.row]);
        }

        // Every latent row and column is written by some tile; the seams are written twice, by the
        // tile that owns them and by the neighbour that fades into them.
        let mut rows = [0u32; 48];
        let mut columns = [0u32; 84];
        for tile in &plan.tiles {
            let (start, length) = tile.rows;
            for row in start..start + length {
                rows[row as usize] += 1;
            }
            let (start, length) = tile.columns;
            for column in start..start + length {
                columns[column as usize] += 1;
            }
        }
        assert!(rows.iter().all(|count| *count >= 1));
        assert!(columns.iter().all(|count| *count >= 1));
    }

    #[test]
    fn a_seam_fades_over_the_overlap_the_split_was_given() {
        let encoder = canvas();
        // The split spreads the 64 pixels of slack over the seams in tile order, in whole
        // 16-pixel steps; the tall axis has three seams and the wide axis six.
        assert_eq!(encoder.grid().rows.overlaps, vec![96, 80, 80]);
        assert_eq!(Grid::blends(&encoder.grid().rows), vec![6, 5, 5]);
        assert_eq!(
            encoder.grid().columns.overlaps,
            vec![80, 80, 80, 80, 64, 64]
        );
        assert_eq!(
            Grid::blends(&encoder.grid().columns),
            vec![5, 5, 5, 5, 4, 4]
        );

        let plan = encoder.plan();
        // The raster's first tile row and column have nothing to fade against; every other tile
        // fades over the seam it shares with the neighbour before it.
        assert_eq!(plan.tiles[0].above, 0);
        assert_eq!(plan.tiles[0].left, 0);
        assert_eq!(plan.tiles[1].above, 0);
        assert_eq!(plan.tiles[1].left, 5);
        assert_eq!(plan.tiles[7].above, 6);
        assert_eq!(plan.tiles[8].above, 6);
        assert_eq!(plan.tiles[8].left, 5);
        // The last column's fade is its own seam, not the rows': the two axes have different
        // widths, and reading one for the other is a defect this assertion is here to catch.
        assert_eq!(plan.tiles[6].left, 4);
        assert_eq!(plan.tiles[6].above, 0);
    }

    #[test]
    fn a_band_reads_the_rows_its_latents_depend_on_and_crops_them_back() {
        let encoder = canvas();
        let whole = encoder.plan();

        // A band starting inside the first seam reads the tile row above it, because those rows
        // cross-fade with it.
        let seam = encoder.band(4, 12).expect("a band inside the first seam");
        assert_eq!(seam.origin, 0);
        assert_eq!(seam.crop, (4, 12));
        assert_eq!(seam.tiles.first().map(|tile| tile.row), Some(0));
        assert_eq!(seam.tiles.last().map(|tile| tile.row), Some(1));
        assert!(!seam.tiles[0].fades_above());

        // A band whose first row is past a seam reads the tile row that owns it and nothing above.
        let owned = encoder.band(20, 40).expect("a band on a tile row");
        assert_eq!(owned.origin, 10);
        assert_eq!(owned.tiles.first().map(|tile| tile.row), Some(1));
        assert_eq!(owned.tiles.last().map(|tile| tile.row), Some(3));
        assert!(!owned.tiles[0].fades_above());

        // Every tile of a band is a tile of the whole encoding, at the same place.
        for band in [seam, owned] {
            for tile in &band.tiles {
                let same = whole
                    .tiles
                    .iter()
                    .find(|other| other.row == tile.row && other.column == tile.column)
                    .expect("a band's tile is the whole's tile");
                assert_eq!(same.rows, tile.rows);
                assert_eq!(same.columns, tile.columns);
            }
        }
    }

    #[test]
    fn a_band_writes_every_row_it_keeps_and_no_row_it_drops() {
        // The band property at the plan's own level: shifted back by the run's origin, the rows the
        // band's tiles write must be exactly the rows the caller keeps. A band one row short loses a
        // latent row silently, which is why this is asserted row by row rather than as a count.
        let encoder = canvas();
        for (start, stop) in [
            (0, 48),
            (0, 1),
            (4, 12),
            (10, 11),
            (11, 21),
            (20, 40),
            (21, 22),
            (32, 33),
            (47, 48),
        ] {
            let band = encoder.band(start, stop).expect("a band");
            let origin = band.origin;
            let mut written = vec![false; (stop - start) as usize];
            for tile in &band.tiles {
                // The callers' rows, not the run's: `written` covers the crop, and the run starts
                // `start - origin` rows before it.
                let first = tile.rows.0.max(start) - start;
                let last = (tile.rows.0 + tile.rows.1).min(stop) - start;
                for row in first..last {
                    written[row as usize] = true;
                }
            }
            assert!(
                written.iter().all(|row| *row),
                "band [{start}, {stop}) from origin {origin} leaves a row unwritten"
            );
        }
    }

    #[test]
    fn the_bands_of_a_raster_cover_its_rows_without_gap_or_overlap() {
        let encoder = canvas();
        let mut next = 0;
        for stop in [1, 4, 10, 11, 21, 22, 32, 33, 47, 48] {
            let band = encoder.band(next, stop).expect("a band");
            assert_eq!(band.crop, (next, stop));
            assert!(band.origin <= next);
            assert!(band.extent().1 >= stop);
            next = stop;
        }
        assert_eq!(next, 48);
        assert!(encoder.band(48, 48).is_err());
        assert!(encoder.band(0, 49).is_err());
    }
}

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

use crate::encoding::{Grid, SPATIAL_RATIO};
use crate::{Error, Result};

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
    /// The latent rows the run covers before the crop, `(origin, extent)`.
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

/// The tiled encoding of one raster: the tile grid, and the band arithmetic over it.
///
/// The raster's extents are input pixels and must be multiples of the spatial compression, which is
/// what [`Grid::of`] checks; the latents are the raster divided by it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ClipEncoder {
    grid: Grid,
    latent: (u32, u32),
}

impl ClipEncoder {
    /// The tiled encoding of a `height` by `width` raster, in input pixels.
    pub fn new(height: u32, width: u32) -> Result<Self> {
        let grid = Grid::of(height, width)?;
        Ok(Self {
            grid,
            latent: (height / SPATIAL_RATIO, width / SPATIAL_RATIO),
        })
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
        // A raster inside one tile has no seam at all.
        let small = ClipEncoder::new(256, 256).expect("one tile");
        assert!(small.blend_extents().is_empty());
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
        assert_eq!(rows.iter().copied().min(), Some(1));
        assert_eq!(columns.iter().copied().min(), Some(1));
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

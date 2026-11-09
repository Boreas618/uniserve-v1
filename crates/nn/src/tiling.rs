//! Tiled VAE execution: the overlapping tile grid of a raster, the blends that hide the seams, the
//! temporal clips a long video is decoded in, and the encode and decode drivers over them.
//!
//! ## The grid
//!
//! [`Seams::split`] is the reference's `split_tiles` along one axis: the fewest tiles of `tile`
//! samples whose minimum-`overlap` coverage reaches the extent, every tile the same length, and the
//! slack beyond the minimum spread over the seams in tile order, one alignment step each,
//! cyclically. A [`Grid`] splits both axes; its [`Tile`]s are row-major, the order the reference
//! encodes, decodes and blends them, and in the cells of the latent raster (a sample extent over
//! the compression ratio).
//!
//! ## The blend
//!
//! One ramp hides every seam: position `i` of an `extent`-wide overlap reads `1 - i/extent` of the
//! previous tile and `i/extent` of the current one, with the ramp, its complement, both products and
//! their sum each rounded to the blended element type, as the reference's `blend(a, b, extent, dim)`
//! rounds them. Its forms differ in where the roundings fall when two seams cross or when the
//! overlap runs along frames, and each consumer keeps the form its served outputs were produced
//! with (measured on device, `artifacts/model-library/equivalence/RESULTS.md`):
//!
//! - [`Blend`] along rows or columns is one `cross_fade` launch;
//! - [`Blend`] along frames is the stepwise rule: per channel and frame, the previous frame times
//!   the complement, the current frame times the weight, and their sum, three rounded `axpby`
//!   launches (it differs from `cross_fade` by up to 2048 ulp);
//! - [`TiledDecode`]'s stitch is the fused corner rule: a raster pixel reads its own tile's token
//!   and the tokens of the tile above and to the left as decoded, in one `tile_stitch` launch (its
//!   edges and interior equal vertical-then-horizontal [`Blend`]s; its corners differ).
//!
//! ## The drivers
//!
//! [`TiledEncode`] runs a model's per-tile forward over a grid and stitches the results, sharing the
//! tiles among the ranks of the forward's split in contiguous shares and gathering them. Every rank
//! then stitches all of them, so the stitched raster is the same on every rank. [`TiledDecode`]
//! runs a model's per-batch forward over every tile at once and stitches its tokens into the raster;
//! its tiles are not shared among ranks, because a share changes every batched product's row count
//! and so possibly the kernel a product selects, which is not provably the served arithmetic.

use ops::tile_stitch::{self, TileStitch};
use ops::{DType, Shape, Strides, TensorView};

use crate::capability::Splice;
use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::launch::PDL;
use crate::layers::{Access, Axpby, Copy};

/// One axis split into overlapping tiles, in samples of the axis.
///
/// `overlaps[i]` is the overlap between tile `i` and tile `i + 1`. Every value is a multiple of the
/// split's alignment.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Seams {
    /// Where each tile starts.
    pub starts: Vec<u32>,
    /// How long each tile is: the tile length, or the whole extent when one tile covers it.
    pub lengths: Vec<u32>,
    /// How much tile `i` overlaps tile `i + 1`.
    pub overlaps: Vec<u32>,
}

impl Seams {
    /// Split `length` samples into tiles of `tile` with at least `overlap` shared, on a grid of
    /// `alignment` samples.
    ///
    /// # Errors
    ///
    /// A zero extent, tile or alignment, an overlap no narrower than the tile, or an extent, tile or
    /// overlap that is not a multiple of the alignment.
    pub fn split(length: u32, tile: u32, overlap: u32, alignment: u32) -> Result<Self> {
        const OWNER: &str = "Seams";
        if length == 0 || tile == 0 || alignment == 0 {
            return Err(Error::shape(
                OWNER,
                "a tile split needs a positive extent, tile and alignment".to_owned(),
            ));
        }
        if overlap >= tile {
            return Err(Error::shape(
                OWNER,
                format!("an overlap of {overlap} does not fit a tile of {tile}"),
            ));
        }
        for (name, value) in [("extent", length), ("tile", tile), ("overlap", overlap)] {
            if !value.is_multiple_of(alignment) {
                return Err(Error::shape(
                    OWNER,
                    format!("the {name} {value} is not a multiple of the alignment {alignment}"),
                ));
            }
        }
        if tile >= length {
            return Ok(Self {
                starts: vec![0],
                lengths: vec![length],
                overlaps: Vec::new(),
            });
        }

        // The fewest tiles whose minimum-overlap coverage reaches the extent.
        let mut count = length.div_ceil(tile);
        while u64::from(tile) * u64::from(count) - u64::from(overlap) * u64::from(count - 1)
            < u64::from(length)
        {
            count += 1;
        }

        // The slack beyond the minimum overlap, spread over the seams in tile order in whole
        // alignment steps: the first seams of the grid take one step each, cyclically.
        let seams = count - 1;
        let mut overlaps = vec![overlap; seams as usize];
        let covered = u64::from(tile) * u64::from(count) - u64::from(overlap) * u64::from(seams);
        let slack = covered - u64::from(length);
        for index in 0..(slack / u64::from(alignment)) as usize {
            let seam = index % overlaps.len();
            overlaps[seam] += alignment;
        }

        let mut starts = Vec::with_capacity(count as usize);
        starts.push(0);
        for overlap in &overlaps {
            let next = starts[starts.len() - 1] + tile - overlap;
            starts.push(next);
        }
        Ok(Self {
            starts,
            lengths: vec![tile; count as usize],
            overlaps,
        })
    }

    /// How many tiles this axis has.
    pub fn len(&self) -> usize {
        self.starts.len()
    }

    /// Whether this axis has no tile.
    pub fn is_empty(&self) -> bool {
        self.starts.is_empty()
    }

    /// The overlap between tile `index` and tile `index + 1`, if there is a tile after it.
    pub fn overlap(&self, index: usize) -> Option<u32> {
        self.overlaps.get(index).copied()
    }

    /// The rows of the stitch table of this axis over `length` samples: for each position, the tile
    /// that keeps it, the position inside that tile, and the width of the seam the position lies in
    /// (zero outside one).
    ///
    /// A tile keeps everything up to where the next one starts, the reference's crop of its
    /// trailing overlap; a position in the first `overlap` of a tile lies in the seam with its
    /// predecessor.
    fn stitch(&self, length: u32) -> Vec<[i32; 3]> {
        let mut rows = Vec::with_capacity(length as usize);
        let mut tile = 0_usize;
        for position in 0..length {
            while tile + 1 < self.len() && self.starts[tile + 1] <= position {
                tile += 1;
            }
            let inside = position - self.starts[tile];
            let seam = tile
                .checked_sub(1)
                .and_then(|previous| self.overlap(previous))
                .filter(|overlap| inside < *overlap)
                .unwrap_or(0);
            rows.push([tile as i32, inside as i32, seam as i32]);
        }
        rows
    }
}

/// One tile of a [`Grid`], in cells of the latent raster.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Tile {
    /// Row of the tile grid.
    pub row: usize,
    /// Column of the tile grid.
    pub column: usize,
    /// The tile's first row and its row count.
    pub rows: (u32, u32),
    /// The tile's first column and its column count.
    pub columns: (u32, u32),
    /// The rows it shares with the tile above, which it fades over; zero on the first tile row.
    pub above: u32,
    /// The columns it shares with the tile to its left; zero on the first tile column.
    pub left: u32,
    /// The rows the tile below shares with it; zero on the last tile row.
    pub below: u32,
    /// The columns the tile to its right shares with it; zero on the last tile column.
    pub right: u32,
}

/// A raster split into overlapping tiles along both spatial axes, in samples of the raster.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Grid {
    /// The tile rows.
    pub rows: Seams,
    /// The tile columns.
    pub columns: Seams,
}

impl Grid {
    /// The grid of a `height` by `width` raster: tiles of `tile` samples square with at least
    /// `overlap` shared, aligned to `alignment`.
    ///
    /// # Errors
    ///
    /// As [`Seams::split`] on either axis.
    pub fn split(height: u32, width: u32, tile: u32, overlap: u32, alignment: u32) -> Result<Self> {
        Ok(Self {
            rows: Seams::split(height, tile, overlap, alignment)?,
            columns: Seams::split(width, tile, overlap, alignment)?,
        })
    }

    /// How many tiles the grid has.
    pub fn len(&self) -> usize {
        self.rows.len() * self.columns.len()
    }

    /// Whether the grid has no tile.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The extent of every tile, `(rows, columns)` in cells of `ratio` samples.
    ///
    /// A split keeps every tile of an axis the same length, so one extent is every tile's.
    pub fn tile(&self, ratio: u32) -> (u32, u32) {
        let first = |seams: &Seams| seams.lengths.first().copied().unwrap_or(0) / ratio.max(1);
        (first(&self.rows), first(&self.columns))
    }

    /// Every tile, row-major, in cells of `ratio` samples: the order the tiles are run and
    /// blended in.
    pub fn tiles(&self, ratio: u32) -> Vec<Tile> {
        let ratio = ratio.max(1);
        let cells = |seams: &Seams, index: usize| {
            (
                seams.starts[index] / ratio,
                seams.lengths[index] / ratio,
                index
                    .checked_sub(1)
                    .and_then(|previous| seams.overlap(previous))
                    .unwrap_or(0)
                    / ratio,
                seams.overlap(index).unwrap_or(0) / ratio,
            )
        };
        let mut tiles = Vec::with_capacity(self.len());
        for row in 0..self.rows.len() {
            let (top, rows, above, below) = cells(&self.rows, row);
            for column in 0..self.columns.len() {
                let (first, columns, left, right) = cells(&self.columns, column);
                tiles.push(Tile {
                    row,
                    column,
                    rows: (top, rows),
                    columns: (first, columns),
                    above,
                    left,
                    below,
                    right,
                });
            }
        }
        tiles
    }

    /// The grid's stitch table over a `height` by `width` raster: the rows axis's rows, then the
    /// columns axis's, `[tile, position inside it, seam width]` each, as the decode stitch reads it.
    pub fn stitch_table(&self, height: u32, width: u32) -> Vec<[i32; 3]> {
        let mut table = self.rows.stitch(height);
        table.extend(self.columns.stitch(width));
        table
    }
}

/// The voxel index of every row of a tiled block: each tile's own cells, frame by frame, tile after
/// tile, so a voxel in an overlap has one row per tile that holds it.
///
/// `width` is the raster's columns; a voxel's index is `(frame · height + row) · width + column`.
pub fn voxels(tiles: &[Tile], frames: u32, height: u32, width: u32) -> Vec<u32> {
    let mut rows = Vec::new();
    for tile in tiles {
        for frame in 0..frames {
            for row in tile.rows.0..tile.rows.0 + tile.rows.1 {
                for column in tile.columns.0..tile.columns.0 + tile.columns.1 {
                    rows.push((frame * height + row) * width + column);
                }
            }
        }
    }
    rows
}

/// Gather the tiles' cells of `frames` frames from frame `first` of a channel-major `trajectory`
/// into `block`, one row per voxel of [`voxels`]' order and one column per channel.
///
/// `trajectory` holds `channels` planes of `[frames', height, width]` cells, each plane one whole
/// timeline; `block` is `[voxels, channels]` of the same element type. One strided copy per tile.
///
/// # Errors
///
/// A clip past the trajectory, views of other types, or the copy's refusal.
#[allow(
    clippy::too_many_arguments,
    reason = "the trajectory's geometry and the clip's window are separate statements"
)]
pub fn gather(
    f: &Forward,
    tiles: &[Tile],
    trajectory: TensorView,
    channels: u32,
    (height, width): (u32, u32),
    first: u32,
    frames: u32,
    block: TensorView,
) -> Result<()> {
    let channels = u64::from(channels);
    let (height, width) = (u64::from(height), u64::from(width));
    let plane = height * width;
    let voxels = trajectory.numel() / channels.max(1);
    if channels == 0
        || !trajectory.numel().is_multiple_of(channels)
        || (u64::from(first) + u64::from(frames)) * plane > voxels
        || block.dtype != trajectory.dtype
    {
        return Err(Error::shape(
            "gather",
            format!(
                "a clip of {frames} frames from frame {first} of a {plane}-cell raster is past a \
                 trajectory of {} {} elements in {channels} channels, or a block of {}",
                trajectory.numel(),
                trajectory.dtype,
                block.dtype
            ),
        ));
    }
    let copy = Copy::new(trajectory.dtype, Access::Element)?;
    let frames = u64::from(frames);
    let mut row = 0_u64;
    for tile in tiles {
        let (top, rows) = (u64::from(tile.rows.0), u64::from(tile.rows.1));
        let (left, columns) = (u64::from(tile.columns.0), u64::from(tile.columns.1));
        let shape = Shape::new(&[channels, frames, rows, columns]);
        let start = u64::from(first) * plane + top * width + left;
        let source = TensorView::with_strides(
            trajectory.offset_elements(start),
            trajectory.dtype,
            shape,
            Strides::new(&[voxels, plane, width, 1]),
        );
        let strides = [1, rows * columns * channels, columns * channels, channels];
        let destination = TensorView::with_strides(
            block.offset_elements(row * channels),
            block.dtype,
            shape,
            Strides::new(&strides),
        );
        copy.forward(f, source, destination, &strides.map(|stride| stride as i64))?;
        row += frames * rows * columns;
    }
    Ok(())
}

/// The axis a [`Blend`] runs along, of a `[channels, frames, rows, columns]` raster.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Axis {
    /// Along the frames: a clip fading in against its predecessor's tail, stepwise.
    Frames,
    /// Along the rows: a tile fading in against the tile above it.
    Rows,
    /// Along the columns: a tile fading in against the tile to its left.
    Columns,
}

/// The overlap blend of two `[channels, frames, rows, columns]` blocks along one axis.
///
/// The leading `extent` positions of the axis blend the previous block's last `extent` positions
/// with the current block's first; every later position is the current block's. Along rows or
/// columns it is one `cross_fade` launch over views of any strides; along frames it is the stepwise
/// rule (see the module documentation), whose operands are one contiguous frame per channel.
#[derive(Clone, Copy, Debug)]
pub struct Blend {
    dtype: DType,
    extent: u32,
    axis: Axis,
}

impl Blend {
    /// The blend of `dtype` blocks over `extent` positions of `axis`.
    ///
    /// # Errors
    ///
    /// An overlap of no positions.
    pub fn new(dtype: DType, extent: u32, axis: Axis) -> Result<Self> {
        if extent == 0 {
            return Err(Error::shape(
                "Blend",
                "an overlap of no positions blends nothing".to_owned(),
            ));
        }
        Ok(Self {
            dtype,
            extent,
            axis,
        })
    }

    /// The blend's extent along its axis.
    pub fn extent(&self) -> u32 {
        self.extent
    }

    /// The blend's axis.
    pub fn axis(&self) -> Axis {
        self.axis
    }

    /// Blend `previous` and `current` into `out`.
    ///
    /// Along rows or columns the three views share one shape and `out` may be `current`. Along
    /// frames, `previous` is the predecessor's tail of `extent` frames, `current` the clip whose
    /// first `extent` frames fade in, and `out` the `extent` blended frames, all contiguous per
    /// frame of a channel.
    ///
    /// # Errors
    ///
    /// Views the blend's operators refuse, or frame operands of another channel count or extent.
    pub fn forward(
        &self,
        f: &Forward,
        previous: TensorView,
        current: TensorView,
        out: TensorView,
    ) -> Result<()> {
        match self.axis {
            Axis::Rows | Axis::Columns => {
                let descriptor = ops::cross_fade::Descriptor {
                    dtype: self.dtype,
                    extent: self.extent,
                    axis: match self.axis {
                        Axis::Rows => 2,
                        _ => 3,
                    },
                    block: ops::cross_fade::BLOCK,
                    pdl: PDL,
                };
                f.operate::<ops::cross_fade::CrossFade>(
                    &descriptor,
                    ops::cross_fade::Args {
                        previous,
                        current,
                        out,
                    },
                )
            }
            Axis::Frames => self.frames(f, previous, current, out),
        }
    }

    /// The stepwise frame rule: per channel and frame `i`, `out = previous · (1 - w) + current · w`
    /// with `w = i / extent`, each of the ramp, its complement, both products and the sum rounded
    /// to the element type, as three `axpby` launches.
    fn frames(
        &self,
        f: &Forward,
        previous: TensorView,
        current: TensorView,
        out: TensorView,
    ) -> Result<()> {
        const OWNER: &str = "Blend";
        let extent = u64::from(self.extent);
        let channels = out.shape.dim(0);
        let pixels = out.shape.dim(2) * out.shape.dim(3);
        if out.shape.rank() != 4
            || out.shape.dims() != [channels, extent, out.shape.dim(2), out.shape.dim(3)]
            || previous.shape != out.shape
            || current.shape.rank() != 4
            || current.shape.dim(0) != channels
            || current.shape.dim(1) < extent
            || current.shape.dim(2) * current.shape.dim(3) != pixels
        {
            return Err(Error::shape(
                OWNER,
                format!(
                    "a {extent}-frame fade of {:?} against {:?} into {:?}",
                    current.shape.dims(),
                    previous.shape.dims(),
                    out.shape.dims()
                ),
            ));
        }
        let round = |value: f32| -> Result<f32> {
            let bytes = self.dtype.encode(&[value]).map_err(Error::from)?;
            Ok(self.dtype.decode(&bytes).map_err(Error::from)?[0])
        };
        let frame = |view: TensorView, channel: u64, index: u64| {
            TensorView::new(
                view.element(&[channel, index, 0, 0]),
                view.dtype,
                Shape::new(&[1, pixels]),
            )
        };
        // The current frame's weighted term, one frame of scratch reused for every frame.
        let mark = f.arena().mark();
        let term = f.arena().tensor(out.dtype, Shape::new(&[1, pixels]));
        let result = term.and_then(|term| {
            for channel in 0..channels {
                for index in 0..extent {
                    let weight = round(index as f32 / extent as f32)?;
                    let complement = round(1.0 - weight)?;
                    let target = frame(out, channel, index);
                    // A zero `beta` does not read the target, so neither buffer needs clearing.
                    Axpby.forward(f, target, frame(previous, channel, index), complement, 0.0)?;
                    Axpby.forward(f, term, frame(current, channel, index), weight, 0.0)?;
                    Axpby.forward(f, target, term, 1.0, 1.0)?;
                }
            }
            Ok(())
        });
        f.arena().release(mark);
        result
    }
}

/// The temporal clips a long video is decoded in: each clip decodes a window of latent frames, keeps
/// a body, and hands an overlap tail to its successor, which fades its first frames against it.
///
/// A timeline of `n` clips is `body · n + overlap` frames; which frame counts are timelines is the
/// family's frame rule, which states the clips a count holds. Clip `k` reads `window()` latent frames
/// from `k · advance`, decodes them to `window() · temporal` frames of which the leading surplus
/// over `native()` is cropped, keeps its first `body` frames, skips `padding`, and its last
/// `overlap` frames are the tail; the last clip keeps its tail.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TemporalClips {
    /// Frames a clip keeps before its tail.
    pub body: u32,
    /// Frames between the body and the tail that no clip keeps.
    pub padding: u32,
    /// Frames of a clip's tail, which its successor fades its first frames against.
    pub overlap: u32,
    /// Decoded frames per latent frame.
    pub temporal: u32,
    /// Latent frames between one clip's window and the next's.
    pub advance: u32,
}

impl TemporalClips {
    /// The frames of a clip's own decode after the crop: body, padding and tail.
    pub fn native(&self) -> u32 {
        self.body + self.padding + self.overlap
    }

    /// The latent frames a clip's window reads: `ceil(native / temporal)`.
    pub fn window(&self) -> u32 {
        self.native().div_ceil(self.temporal.max(1))
    }

    /// The first latent frame of clip `clip`'s window.
    pub fn start(&self, clip: u32) -> u32 {
        clip * self.advance
    }

    /// The frames of a clip's decode its tail occupies, `(start, end)`.
    pub fn tail(&self) -> (u32, u32) {
        (self.body + self.padding, self.native())
    }

    /// The splices a timeline of `count` clips is written in: clip `i` writes from frame
    /// `body · i`, every clip but the first fades `overlap` frames against its predecessor, and the
    /// last keeps its tail.
    ///
    /// # Errors
    ///
    /// A timeline of no clip.
    pub fn splices(&self, count: u32) -> Result<Vec<Splice>> {
        let (body, tail) = (self.body, self.overlap);
        if count == 0 {
            return Err(Error::shape(
                "TemporalClips",
                "a timeline holds at least one clip".to_owned(),
            ));
        }
        Ok((0..count)
            .map(|index| Splice {
                keep: body + if index + 1 == count { tail } else { 0 },
                overlap: if index == 0 { 0 } else { tail },
                first: index * body,
            })
            .collect())
    }
}

/// The device buffers one [`TiledEncode`] run works in, reserved once for all its clips.
#[derive(Clone, Copy)]
pub struct TileBlocks {
    /// This rank's share of the tiles' blocks.
    own: TensorView,
    /// Every rank's share, in tile order; `own` itself without a split.
    gathered: TensorView,
    /// The block a two-way fade hands from its vertical blend to its horizontal one.
    handed: TensorView,
    rank: usize,
    ranks: usize,
    /// The tiles of one rank's share.
    share: usize,
}

/// A tiled encode: a model's per-tile forward over a grid, its results shared among the forward's
/// ranks and stitched into one raster.
///
/// Each tile's block fades over the seam it shares with the tile above and the tile to its left,
/// both read as the model wrote them, never as already blended: that is what makes the overlap
/// where two seams cross follow the arithmetic of the seams themselves. A tile that fades both ways
/// blends vertically into a hand-off block and then horizontally into the raster.
pub struct TiledEncode {
    grid: Grid,
    tiles: Vec<Tile>,
    latent: (u32, u32),
    /// One compiled blend per seam width and axis.
    blends: Vec<Blend>,
    /// How the run's first tile reaches the raster; see [`Self::place_first`].
    first: Blend,
}

impl TiledEncode {
    /// The tiled encode of a `height` by `width` raster in tiles of `tile` samples with at least
    /// `overlap` shared, compressed `ratio`-fold, blending in `dtype`.
    ///
    /// # Errors
    ///
    /// A raster, tile or overlap the split refuses (each a multiple of `ratio`).
    pub fn new(
        height: u32,
        width: u32,
        tile: u32,
        overlap: u32,
        ratio: u32,
        dtype: DType,
    ) -> Result<Self> {
        let grid = Grid::split(height, width, tile, overlap, ratio)?;
        let tiles = grid.tiles(ratio);
        let mut blends: Vec<Blend> = Vec::new();
        for tile in &tiles {
            for (extent, axis) in [(tile.above, Axis::Rows), (tile.left, Axis::Columns)] {
                if extent != 0
                    && !blends
                        .iter()
                        .any(|blend| blend.extent == extent && blend.axis == axis)
                {
                    blends.push(Blend::new(dtype, extent, axis)?);
                }
            }
        }
        Ok(Self {
            grid,
            tiles,
            latent: (height / ratio, width / ratio),
            blends,
            first: Blend::new(dtype, 1, Axis::Rows)?,
        })
    }

    /// The grid, in samples of the raster.
    pub fn grid(&self) -> &Grid {
        &self.grid
    }

    /// Every tile, row-major, in latent cells.
    pub fn tiles(&self) -> &[Tile] {
        &self.tiles
    }

    /// The latent raster the stitch writes, `(rows, columns)`.
    pub fn latent(&self) -> (u32, u32) {
        self.latent
    }

    /// The blends a run launches, one per seam width and axis.
    pub fn blends(&self) -> &[Blend] {
        &self.blends
    }

    /// The `[channels, frames, rows, columns]` block of one tile.
    fn block(&self, channels: u64, frames: u64) -> Shape {
        let tile = self.tiles.first().copied();
        let (rows, columns) = tile.map_or((0, 0), |tile| (tile.rows.1, tile.columns.1));
        Shape::new(&[channels, frames, u64::from(rows), u64::from(columns)])
    }

    /// Reserve the blocks a run of clips of at most `frames` latent frames and `channels` channels
    /// encodes into, in `dtype`.
    ///
    /// The tiles are shared among the forward's ranks in contiguous shares, `ceil(tiles / ranks)`
    /// each: the gather lays the shares end to end, so rank `r`'s tiles are the tiles from
    /// `r · share`.
    ///
    /// # Errors
    ///
    /// The arena's refusal.
    pub fn reserve(
        &self,
        f: &Forward,
        dtype: DType,
        channels: u64,
        frames: u64,
    ) -> Result<TileBlocks> {
        let (rank, ranks) = f
            .split()
            .map_or((0, 1), |split| (split.rank as usize, split.ranks as usize));
        let share = self.tiles.len().div_ceil(ranks);
        let block = self.block(channels, frames);
        let stack = |tiles: usize| {
            let dims = block.dims();
            Shape::new(&[tiles as u64, dims[0], dims[1], dims[2], dims[3]])
        };
        let own = f.arena().tensor(dtype, stack(share))?;
        let gathered = match ranks {
            1 => own,
            _ => f.arena().tensor(dtype, stack(share * ranks))?,
        };
        Ok(TileBlocks {
            own,
            gathered,
            handed: f.arena().tensor(dtype, block)?,
            rank,
            ranks,
            share,
        })
    }

    /// Run `encode` over this rank's tiles and stitch every rank's into `out`.
    ///
    /// `encode(index, block)` writes tile `index`'s `[channels, frames, rows, columns]` block,
    /// contiguous, where `frames` and `channels` are `out`'s; `out` is the
    /// `[channels, frames, latent rows, latent columns]` raster, of any strides. The tiles are
    /// gathered over the forward's split when it has more than one rank.
    ///
    /// # Errors
    ///
    /// A raster of another rank or type or one smaller than the latent raster, blocks reserved for
    /// fewer frames, a collective without a group, or the forward's or an operator's refusal.
    pub fn forward(
        &self,
        f: &Forward,
        blocks: &TileBlocks,
        mut encode: impl FnMut(usize, TensorView) -> Result<()>,
        out: TensorView,
    ) -> Result<()> {
        const OWNER: &str = "TiledEncode";
        if out.shape.rank() != 4
            || out.dtype != blocks.own.dtype
            || out.shape.dim(2) < u64::from(self.latent.0)
            || out.shape.dim(3) < u64::from(self.latent.1)
        {
            return Err(Error::shape(
                OWNER,
                format!(
                    "a {} raster of {:?} does not hold the {:?} latent cells in {}",
                    out.dtype,
                    out.shape.dims(),
                    self.latent,
                    blocks.own.dtype
                ),
            ));
        }
        let block = self.block(out.shape.dim(0), out.shape.dim(1));
        let elements = block.numel();
        if elements * blocks.share as u64 > blocks.own.numel() {
            return Err(Error::shape(
                OWNER,
                format!(
                    "blocks of {:?} do not fit the reserved {:?}",
                    block.dims(),
                    blocks.own.shape.dims()
                ),
            ));
        }
        // A clip's blocks lie one after another at this clip's own size: a last clip may be shorter
        // than the run's widest.
        let block_of = |stack: TensorView, index: usize| {
            stack
                .at(stack.offset_elements(index as u64 * elements))
                .reshape(block)
        };
        let count = self.tiles.len();
        let mine =
            (blocks.rank * blocks.share).min(count)..((blocks.rank + 1) * blocks.share).min(count);
        for index in mine.clone() {
            encode(index, block_of(blocks.own, index - mine.start))?;
        }
        if blocks.ranks > 1 {
            // This clip's blocks alone: a member's share of them lies at the head of its own.
            let shared = blocks.share as u64 * elements;
            f.all_gather(
                blocks.own.reshape(Shape::new(&[shared])),
                blocks
                    .gathered
                    .reshape(Shape::new(&[shared * blocks.ranks as u64])),
            )?;
        }

        // Row-major tiles: the neighbour above is one grid row back, the one to the left the tile
        // before.
        let columns = self.grid.columns.len();
        let handed = blocks.handed.reshape(block);
        for (index, tile) in self.tiles.iter().enumerate() {
            let decoded = block_of(blocks.gathered, index);
            let placed = place(out, tile, block)?;
            let above = (tile.above != 0).then(|| block_of(blocks.gathered, index - columns));
            let left = (tile.left != 0).then(|| block_of(blocks.gathered, index - 1));
            match (above, left) {
                (Some(above), Some(left)) => {
                    self.blend(tile.above, Axis::Rows)?
                        .forward(f, above, decoded, handed)?;
                    self.blend(tile.left, Axis::Columns)?
                        .forward(f, left, handed, placed)?;
                }
                (Some(above), None) => {
                    self.blend(tile.above, Axis::Rows)?
                        .forward(f, above, decoded, placed)?;
                }
                (None, Some(left)) => {
                    self.blend(tile.left, Axis::Columns)?
                        .forward(f, left, decoded, placed)?;
                }
                (None, None) => self.place_first(f, decoded, placed)?,
            }
        }
        Ok(())
    }

    /// Place the run's first tile, which has no neighbour to fade against.
    ///
    /// The served encode places it with a one-position row blend of the tile with itself, and that
    /// blend reads the previous source at `rows - extent + i`: at position 0 it writes
    /// `tile[rows - 1] · 1 + tile[0] · 0`, so the tile's **last** row lands in its first, and every
    /// later row is copied. The reference copies the tile unchanged. This form is kept because the
    /// served outputs carry it; correcting it changes every conditioned encode and is a protocol
    /// change of its own.
    fn place_first(&self, f: &Forward, tile: TensorView, out: TensorView) -> Result<()> {
        self.first.forward(f, tile, tile, out)
    }

    /// The compiled blend of `extent` along `axis`.
    fn blend(&self, extent: u32, axis: Axis) -> Result<Blend> {
        self.blends
            .iter()
            .find(|blend| blend.extent == extent && blend.axis == axis)
            .copied()
            .ok_or_else(|| {
                Error::shape(
                    "TiledEncode",
                    format!("no {extent}-position blend along {axis:?} was compiled"),
                )
            })
    }
}

/// A view of `raster` holding `tile`'s block at the tile's place, at the raster's strides.
fn place(raster: TensorView, tile: &Tile, block: Shape) -> Result<TensorView> {
    let (rows, columns) = (u64::from(tile.rows.0), u64::from(tile.columns.0));
    if rows + block.dim(2) > raster.shape.dim(2) || columns + block.dim(3) > raster.shape.dim(3) {
        return Err(Error::shape(
            "TiledEncode",
            format!(
                "the block at ({rows}, {columns}) of {:?} does not fit {:?}",
                block.dims(),
                raster.shape.dims()
            ),
        ));
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

/// A tiled decode: a model's per-batch forward over every tile at once, its tokens stitched into the
/// raster with the fused corner rule (see the module documentation).
///
/// A token holds one `patch_frames × patch_side × patch_side` patch of `channels` channels; a tile
/// is `token_rows` rows of tokens, its patches in raster order followed by any rows the model
/// appends. The stitch reads the grid as the table [`Grid::stitch_table`] writes.
pub struct TiledDecode {
    descriptor: tile_stitch::Descriptor,
    /// Elements of one token.
    patch: u64,
}

impl TiledDecode {
    /// The decode of tokens of `dtype`, each `patch_frames` frames of a `patch_side`-sample square
    /// of `channels` channels.
    ///
    /// # Errors
    ///
    /// No channels or frames, or a side that is not a power of two.
    pub fn new(dtype: DType, channels: u32, patch_frames: u32, patch_side: u32) -> Result<Self> {
        if channels == 0 || patch_frames == 0 || !patch_side.is_power_of_two() {
            return Err(Error::shape(
                "TiledDecode",
                format!(
                    "a patch of {channels} channels, {patch_frames} frames and a side of \
                     {patch_side}"
                ),
            ));
        }
        Ok(Self {
            descriptor: tile_stitch::Descriptor {
                dtype,
                channels,
                patch_frames,
                patch_side,
                block: tile_stitch::BLOCK,
            },
            patch: u64::from(channels)
                * u64::from(patch_frames)
                * u64::from(patch_side)
                * u64::from(patch_side),
        })
    }

    /// Run `decode` over the grid's tiles and stitch its tokens into `out`.
    ///
    /// `decode(tokens)` writes every tile's `token_rows` token rows, tile after tile, into
    /// `[tiles · token_rows, patch]`; `table` is the grid's `[height + width, 3]` i32 stitch table
    /// on the device, and `out` the `[channels, frames, height, width]` raster.
    ///
    /// # Errors
    ///
    /// A table or raster of another extent, the model's error, or the stitch's refusal.
    pub fn forward(
        &self,
        f: &Forward,
        grid: &Grid,
        token_rows: u64,
        decode: impl FnOnce(TensorView) -> Result<()>,
        table: TensorView,
        out: TensorView,
    ) -> Result<()> {
        const OWNER: &str = "TiledDecode";
        let (height, width) = (out.shape.dim(2), out.shape.dim(3));
        if out.shape.rank() != 4
            || out.shape.dim(0) != u64::from(self.descriptor.channels)
            || table.shape.dims() != [height + width, tile_stitch::TABLE_WIDTH]
        {
            return Err(Error::shape(
                OWNER,
                format!(
                    "a stitch table of {:?} for a raster of {:?}",
                    table.shape.dims(),
                    out.shape.dims()
                ),
            ));
        }
        let narrow = |value: u64| {
            u32::try_from(value).map_err(|_| Error::shape(OWNER, format!("{value} exceeds u32")))
        };
        let tokens = f.arena().tensor(
            out.dtype,
            Shape::new(&[token_rows * grid.len() as u64, self.patch]),
        )?;
        decode(tokens)?;
        let (tile_rows, tile_columns) = grid.tile(1);
        let row_table = TensorView::new(
            table.offset_elements(0),
            table.dtype,
            Shape::new(&[height, tile_stitch::TABLE_WIDTH]),
        );
        let column_table = TensorView::new(
            table.offset_elements(height * tile_stitch::TABLE_WIDTH),
            table.dtype,
            Shape::new(&[width, tile_stitch::TABLE_WIDTH]),
        );
        f.operate::<TileStitch>(
            &self.descriptor,
            tile_stitch::Args {
                tokens,
                rows: row_table,
                columns: column_table,
                out,
                tile_columns: narrow(grid.columns.len() as u64)?,
                token_rows: narrow(token_rows)?,
                tile_height: tile_rows,
                tile_width: tile_columns,
            },
        )
    }
}

const _: () = assert!(tile_stitch::TABLE_WIDTH == 3);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_raster_within_one_tile_is_one_tile_with_no_seam() {
        let seams = Seams::split(128, 256, 64, 16).unwrap();
        assert_eq!(seams.starts, vec![0]);
        assert_eq!(seams.lengths, vec![128]);
        assert!(seams.overlaps.is_empty());
        let rows = seams.stitch(128);
        assert!(rows.iter().all(|row| row[0] == 0 && row[2] == 0));
        assert_eq!(rows[127], [0, 127, 0]);
    }

    /// The reference's `split_tiles` on 1344 and 768 pixels: seven and four 256-pixel tiles whose
    /// first seams take the 16-pixel steps of slack, cyclically.
    #[test]
    fn the_slack_spreads_over_the_first_seams_in_alignment_steps() {
        let columns = Seams::split(1344, 256, 64, 16).unwrap();
        assert_eq!(columns.overlaps, vec![80, 80, 80, 80, 64, 64]);
        assert_eq!(columns.starts, vec![0, 176, 352, 528, 704, 896, 1088]);
        assert_eq!(columns.starts.last().unwrap() + 256, 1344);
        let rows = Seams::split(768, 256, 64, 16).unwrap();
        assert_eq!(rows.overlaps, vec![96, 80, 80]);
        assert!(Seams::split(1352, 256, 64, 16).is_err(), "not aligned");
        assert!(
            Seams::split(512, 256, 256, 16).is_err(),
            "an overlap as wide as a tile"
        );
    }

    #[test]
    fn tiles_are_row_major_cells_with_the_overlaps_they_share() {
        let grid = Grid::split(768, 1344, 256, 64, 16).unwrap();
        let tiles = grid.tiles(16);
        assert_eq!(tiles.len(), 4 * 7);
        assert_eq!(grid.tile(16), (16, 16));
        assert_eq!((tiles[0].row, tiles[0].column), (0, 0));
        assert_eq!((tiles[1].row, tiles[1].column), (0, 1));
        assert_eq!(tiles[7].rows, (10, 16));
        assert_eq!(tiles[1].columns, (11, 16));
        assert_eq!((tiles[0].above, tiles[0].left), (0, 0));
        assert_eq!((tiles[8].above, tiles[8].left), (6, 5));
        assert_eq!((tiles[0].below, tiles[0].right), (6, 5));
        let last = tiles.last().unwrap();
        assert_eq!((last.below, last.right), (0, 0));
        assert_eq!(last.rows.0 + last.rows.1, 48);
        assert_eq!(last.columns.0 + last.columns.1, 84);
    }

    /// Every position is kept by exactly one tile, which keeps everything up to where the next one
    /// starts, and lies in a seam exactly within the first `overlap` of its tile.
    #[test]
    fn the_stitch_table_keeps_each_position_in_one_tile_and_names_its_seam() {
        let grid = Grid::split(768, 1344, 256, 64, 16).unwrap();
        let table = grid.stitch_table(768, 1344);
        assert_eq!(table.len(), 768 + 1344);
        for (axis, rows, length) in [
            (&grid.rows, &table[..768], 768),
            (&grid.columns, &table[768..], 1344),
        ] {
            for position in 0..length {
                let [tile, inside, seam] = rows[position as usize];
                let tile = tile as usize;
                assert_eq!(axis.starts[tile] + inside as u32, position);
                if let Some(next) = axis.starts.get(tile + 1) {
                    assert!(position < *next);
                }
                let expected = match tile {
                    0 => 0,
                    _ if (inside as u32) < axis.overlaps[tile - 1] => axis.overlaps[tile - 1],
                    _ => 0,
                };
                assert_eq!(seam as u32, expected);
            }
            assert_eq!(rows.last().unwrap()[0] as usize + 1, axis.len());
        }
    }

    #[test]
    fn the_voxels_of_a_tile_are_its_own_cells_frame_by_frame() {
        let grid = Grid::split(32, 48, 32, 16, 16).unwrap();
        let tiles = grid.tiles(16);
        // One tile row of two columns over a 2 x 3 raster: the second tile repeats the seam column.
        assert_eq!(tiles.len(), 2);
        let rows = voxels(&tiles, 2, 2, 3);
        assert_eq!(
            rows,
            vec![0, 1, 3, 4, 6, 7, 9, 10, 1, 2, 4, 5, 7, 8, 10, 11]
        );
    }

    #[test]
    fn a_timeline_is_body_clips_and_one_kept_tail() {
        let clips = TemporalClips {
            body: 17,
            padding: 3,
            overlap: 5,
            temporal: 4,
            advance: 5,
        };
        assert_eq!(clips.native(), 25);
        assert_eq!(clips.window(), 7);
        assert_eq!(clips.start(2), 10);
        assert_eq!(clips.tail(), (20, 25));
        let splices = clips.splices(3).unwrap();
        assert_eq!(splices.len(), 3);
        assert_eq!(
            (splices[0].keep, splices[0].overlap, splices[0].first),
            (17, 0, 0)
        );
        assert_eq!(
            (splices[1].keep, splices[1].overlap, splices[1].first),
            (17, 5, 17)
        );
        assert_eq!(
            (splices[2].keep, splices[2].overlap, splices[2].first),
            (22, 5, 34)
        );
        assert!(clips.splices(0).is_err());
    }
}

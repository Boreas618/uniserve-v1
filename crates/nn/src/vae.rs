//! Tiled VAE execution: the overlapping tiles of a raster, the blend that hides their seams, the
//! stitch of a grid of tiles, the encode that runs a model's forward tile by tile, and the
//! temporal clips a long video is decoded in.
//!
//! ## The tiles
//!
//! [`split_tiles`] is the reference's `split_tiles` along one axis: the fewest tiles of `tile`
//! samples whose minimum-`overlap` coverage reaches the extent, every tile the same length, and
//! the slack beyond the minimum spread over the seams in tile order, one alignment step each,
//! cyclically. An axis is its tiles' ranges; a grid is the ranges of its rows and of its columns,
//! and its tiles are row-major, the order the reference encodes, decodes and blends them in.
//! Two neighbouring ranges overlap by the seam between them.
//!
//! ## The blend
//!
//! One ramp hides every seam ([`blend`]): position `i` of an `extent`-wide overlap reads
//! `1 − i/extent` of the previous tile and `i/extent` of the current one, with the ramp, its
//! complement, both products and their sum each rounded into the blended element type, as the
//! reference's `blend(a, b, extent, dim)` rounds them.
//!
//! ## The stitch
//!
//! [`stitch_tiles`] is diffusers' tile assembly: each tile fades over the seam it shares with
//! the tile above and then over the seam with the tile to its left, both read as the model
//! wrote them and never as already blended, which is what makes a corner follow the arithmetic
//! of the seams themselves; a tile then keeps everything up to where the next one starts.
//! An encoder stitches its tiles' latents with it ([`tiled_encode`]) and a decoder its tiles'
//! pixels, each tile unpacked from its tokens.
//!
//! Every function here is a tensor expression, on the CPU and on a device alike: the blend's
//! ramp is a host table moved to the tiles' device.

use std::ops::Range;

use tensor::{DType, Device, Tensor};

use crate::capability::Splice;
use crate::collective;
use crate::error::{Error, Result};

/// One axis of `length` samples split into overlapping tiles of `tile` samples with at least
/// `overlap` shared, on a grid of `alignment` samples: each tile's range, in order.
///
/// An extent no longer than a tile is one tile. Tile `i` overlaps tile `i + 1` by the samples
/// their ranges share.
///
/// # Errors
///
/// A zero extent, tile or alignment, an overlap no narrower than the tile, or an extent, tile
/// or overlap that is not a multiple of the alignment.
pub fn split_tiles(
    length: usize,
    tile: usize,
    overlap: usize,
    alignment: usize,
) -> Result<Vec<Range<usize>>> {
    const OWNER: &str = "split_tiles";

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
        return Ok(std::iter::once(0..length).collect());
    }

    // The fewest tiles whose minimum-overlap coverage reaches the extent.
    let mut count = length.div_ceil(tile);
    while tile * count - overlap * (count - 1) < length {
        count += 1;
    }

    // The slack beyond the minimum overlap, spread over the seams in tile order in whole
    // alignment steps: the first seams take one step each, cyclically.
    let seams = count - 1;
    let mut overlaps = vec![overlap; seams];
    let slack = tile * count - overlap * seams - length;
    for index in 0..slack / alignment {
        overlaps[index % seams] += alignment;
    }

    let mut tiles = Vec::with_capacity(count);
    let mut start = 0;
    tiles.push(start..start + tile);
    for overlap in overlaps {
        start += tile - overlap;
        tiles.push(start..start + tile);
    }
    Ok(tiles)
}

/// The voxel index of every cell of a grid's tiles: each tile's own cells, frame by frame, tile
/// after tile in row-major order, so a voxel in an overlap has one entry per tile that holds
/// it. A CPU `U32` tensor of one axis.
///
/// `rows` and `columns` are the grid's tile ranges in cells of a `height` by `width` raster of
/// `frames` frames; a voxel's index is `(frame · height + row) · width + column`.
///
/// # Errors
///
/// A grid of no tile, or a tile outside the raster.
pub fn tile_voxels(
    rows: &[Range<usize>],
    columns: &[Range<usize>],
    frames: usize,
    height: usize,
    width: usize,
) -> Result<Tensor> {
    let inside = |tiles: &[Range<usize>], extent: usize| {
        !tiles.is_empty() && tiles.iter().all(|tile| tile.end <= extent)
    };
    if !inside(rows, height) || !inside(columns, width) {
        return Err(Error::shape(
            "tile_voxels",
            format!("tile rows {rows:?} and columns {columns:?} of a {height} by {width} raster"),
        ));
    }
    let axis =
        |range: Range<usize>| Tensor::arange(range.start as u32, range.end as u32, &Device::Cpu);
    let scale = |factor: usize| Tensor::new(factor as u32, &Device::Cpu);

    let frame_base = axis(0..frames)?
        .broadcast_mul(&scale(height * width)?)?
        .reshape((frames, 1, 1))?;

    let mut voxels = Vec::with_capacity(rows.len() * columns.len());
    for tile_rows in rows {
        let row_base = axis(tile_rows.clone())?
            .broadcast_mul(&scale(width)?)?
            .reshape((1, tile_rows.len(), 1))?;

        for tile_columns in columns {
            let column = axis(tile_columns.clone())?.reshape((1, 1, tile_columns.len()))?;

            voxels.push(
                frame_base
                    .broadcast_add(&row_base)?
                    .broadcast_add(&column)?
                    .flatten_all()?,
            );
        }
    }
    Ok(Tensor::cat(&voxels, 0)?)
}

/// `x` rounded into `dtype` and widened back: one rounding point of an FP32 formula.
fn rounded(x: &Tensor, dtype: DType) -> Result<Tensor> {
    Ok(x.to_dtype(dtype)?.to_dtype(DType::F32)?)
}

/// The overlap blend of two blocks along `axis`: `current` with its leading `extent` positions
/// faded in against the last `extent` positions of `previous`.
///
/// ```text
/// for i < extent:   out[i] = previous[−extent + i] · (1 − i/extent) + current[i] · (i/extent)
/// elsewhere:        out[i] = current[i]
/// ```
///
/// The ramp `i/extent`, its complement, both products and their sum each round into the
/// blocks' element type. The blocks agree in every extent but `axis`'s, along which each holds
/// at least `extent` positions.
///
/// # Errors
///
/// An overlap of no positions, blocks of two element types, or blocks the overlap does not fit.
pub fn blend(previous: &Tensor, current: &Tensor, extent: usize, axis: usize) -> Result<Tensor> {
    const OWNER: &str = "blend";

    let fits = |block: &Tensor| {
        block
            .dims()
            .get(axis)
            .is_some_and(|length| *length >= extent)
    };
    if extent == 0
        || previous.rank() != current.rank()
        || previous.dtype() != current.dtype()
        || !fits(previous)
        || !fits(current)
    {
        return Err(Error::shape(
            OWNER,
            format!(
                "a {extent}-position fade along axis {axis} of {:?} against {:?}",
                current.dims(),
                previous.dims()
            ),
        ));
    }
    let dtype = current.dtype();
    let (before, length) = (previous.dims()[axis], current.dims()[axis]);

    // The reference's ramp in the element type at every step: `arange(extent) / extent`, then
    // `1 − that`. Both are host tables, shaped to broadcast along the blended axis.
    let mut ramp_shape = vec![1; current.rank()];
    ramp_shape[axis] = extent;
    let positions = rounded(&Tensor::arange(0_f32, extent as f32, &Device::Cpu)?, dtype)?;
    let width = rounded(&Tensor::new(extent as f32, &Device::Cpu)?, dtype)?;
    let ratio = rounded(&positions.broadcast_div(&width)?, dtype)?;
    let complement = rounded(&ratio.affine(-1.0, 1.0)?, dtype)?;
    let on_device = |ramp: Tensor| -> Result<Tensor> {
        Ok(ramp
            .reshape(ramp_shape.as_slice())?
            .to_device(current.device())?)
    };
    let (ratio, complement) = (on_device(ratio)?, on_device(complement)?);

    let faded = |band: Tensor, weight: &Tensor| -> Result<Tensor> {
        rounded(&band.to_dtype(DType::F32)?.broadcast_mul(weight)?, dtype)
    };
    let outgoing = faded(previous.narrow(axis, before - extent, extent)?, &complement)?;
    let incoming = faded(current.narrow(axis, 0, extent)?, &ratio)?;
    let band = (outgoing + incoming)?.to_dtype(dtype)?;

    if length == extent {
        return Ok(band);
    }
    Ok(Tensor::cat(
        &[band, current.narrow(axis, extent, length - extent)?],
        axis,
    )?)
}

/// Checks that `ranges` are an axis's tiles in order, each starting inside its predecessor or
/// where it ends, and returns the extent they cover.
fn covered(owner: &'static str, ranges: &[Range<usize>]) -> Result<usize> {
    let ordered = ranges.windows(2).all(|pair| {
        pair[0].start < pair[1].start && pair[1].start <= pair[0].end && pair[0].end <= pair[1].end
    });

    match (ranges.first(), ranges.last()) {
        (Some(first), Some(last)) if first.start == 0 && ordered => Ok(last.end),
        _ => Err(Error::shape(
            owner,
            format!("{ranges:?} are not the overlapping tiles of an axis from its start"),
        )),
    }
}

/// The raster of a grid of tiles: each tile blended over the seams it shares with the tile
/// above and the tile to its left, cropped to what it keeps, and set in place.
///
/// `tiles` holds the grid's blocks row-major, `rows.len() · columns.len()` of them, block
/// `(r, c)` of the extents `[.., rows[r].len(), columns[c].len()]` on its last two axes;
/// `rows` and `columns` are the tiles' ranges in cells of the raster. A tile fades against its
/// neighbours as the model wrote them ([`blend`], rows first), and keeps its cells up to where
/// the next tile starts; the last tile of an axis keeps all of its own. The result's last two
/// extents are the ranges' ends.
///
/// # Errors
///
/// Ranges that are not an axis's tiles, another count of blocks than the grid, or blocks the
/// ranges do not describe.
pub fn stitch_tiles(
    tiles: &[Tensor],
    rows: &[Range<usize>],
    columns: &[Range<usize>],
) -> Result<Tensor> {
    const OWNER: &str = "stitch_tiles";

    covered(OWNER, rows)?;
    covered(OWNER, columns)?;
    if tiles.len() != rows.len() * columns.len() {
        return Err(Error::shape(
            OWNER,
            format!(
                "{} blocks for a grid of {} by {} tiles",
                tiles.len(),
                rows.len(),
                columns.len()
            ),
        ));
    }

    // The seam a tile shares with its predecessor along an axis, and the cells it keeps.
    let seam = |ranges: &[Range<usize>], index: usize| ranges[index - 1].end - ranges[index].start;
    let kept = |ranges: &[Range<usize>], index: usize| match ranges.get(index + 1) {
        Some(next) => next.start - ranges[index].start,
        None => ranges[index].len(),
    };

    let mut strips = Vec::with_capacity(rows.len());
    for (row, tile_rows) in rows.iter().enumerate() {
        let mut strip = Vec::with_capacity(columns.len());

        for (column, tile_columns) in columns.iter().enumerate() {
            let index = row * columns.len() + column;
            let block = &tiles[index];
            let rank = block.rank();
            if rank < 2 || block.dims()[rank - 2..] != [tile_rows.len(), tile_columns.len()] {
                return Err(Error::shape(
                    OWNER,
                    format!(
                        "tile ({row}, {column}) is {:?}, not rows {tile_rows:?} by columns \
                         {tile_columns:?}",
                        block.dims()
                    ),
                ));
            }
            let (row_axis, column_axis) = (rank - 2, rank - 1);

            let mut blended = block.clone();
            if row > 0 && seam(rows, row) > 0 {
                let above = &tiles[index - columns.len()];
                blended = blend(above, &blended, seam(rows, row), row_axis)?;
            }
            if column > 0 && seam(columns, column) > 0 {
                blended = blend(
                    &tiles[index - 1],
                    &blended,
                    seam(columns, column),
                    column_axis,
                )?;
            }

            strip.push(blended.narrow(row_axis, 0, kept(rows, row))?.narrow(
                column_axis,
                0,
                kept(columns, column),
            )?);
        }
        strips.push(Tensor::cat(&strip, strip[0].rank() - 1)?);
    }
    Ok(Tensor::cat(&strips, strips[0].rank() - 2)?)
}

/// A tiled encode: a model's per-tile forward over a grid, its results shared among the
/// forward's ranks and stitched into one raster ([`stitch_tiles`]).
///
/// `encode(index)` returns tile `index`'s block, row-major over the grid of `rows` and
/// `columns` (the tiles' ranges in cells of the stitched raster): `leading` extents (an
/// encoder's `[channels, frames]`) and then its tile's rows and columns, in `dtype` elements.
/// The tiles are shared among the ranks of the forward's group in contiguous shares,
/// `ceil(tiles / ranks)` each, and gathered when the group has more than one rank, so every
/// rank stitches every tile and the raster is the same on each.
///
/// # Errors
///
/// Ranges that are not an axis's tiles, a block of another extent or type than its tile, a
/// shared grid whose tiles differ in extent, or the model's or a collective's refusal.
pub fn tiled_encode(
    rows: &[Range<usize>],
    columns: &[Range<usize>],
    leading: &[usize],
    dtype: DType,
    mut encode: impl FnMut(usize) -> Result<Tensor>,
) -> Result<Tensor> {
    const OWNER: &str = "tiled_encode";

    covered(OWNER, rows)?;
    covered(OWNER, columns)?;
    let count = rows.len() * columns.len();
    let block_of = |index: usize| {
        let (row, column) = (index / columns.len(), index % columns.len());
        [leading, &[rows[row].len(), columns[column].len()]].concat()
    };

    // A tile's block, checked against the extents and the element type its tile states.
    let mut encoded = |index: usize| -> Result<Tensor> {
        let block = encode(index)?;
        let expected = block_of(index);

        if block.dims() != expected || block.dtype() != dtype {
            return Err(Error::shape(
                OWNER,
                format!(
                    "tile {index} encoded {:?} {}, {expected:?} {} expected",
                    block.dims(),
                    ops::abi::name(block.dtype()),
                    ops::abi::name(dtype)
                ),
            ));
        }
        Ok(block)
    };

    let (rank, ranks) =
        collective::split().map_or((0, 1), |(rank, ranks)| (rank as usize, ranks as usize));
    if ranks == 1 {
        let tiles = (0..count)
            .map(&mut encoded)
            .collect::<Result<Vec<Tensor>>>()?;
        return stitch_tiles(&tiles, rows, columns);
    }

    // The ranks gather equal shares, so the tiles they share are one extent.
    let block = block_of(0);
    if (1..count).any(|index| block_of(index) != block) {
        return Err(Error::shape(
            OWNER,
            "the tiles a group shares are one extent".to_owned(),
        ));
    }

    // This rank's contiguous share of the tiles, stacked. A share past the grid's last tile is
    // gathered with the rest and never stitched, so its blocks are left unwritten.
    let share = count.div_ceil(ranks);
    let mine = (rank * share).min(count)..((rank + 1) * share).min(count);
    let mut own = mine.map(&mut encoded).collect::<Result<Vec<Tensor>>>()?;
    while own.len() < share {
        own.push(backend::empty(block.as_slice(), dtype)?);
    }
    let own = Tensor::stack(&own, 0)?;

    let gathered = backend::empty([&[share * ranks], block.as_slice()].concat(), dtype)?;
    collective::all_gather(&own, &gathered)?;

    let tiles = (0..count)
        .map(|index| Ok(gathered.get(index)?))
        .collect::<Result<Vec<Tensor>>>()?;
    stitch_tiles(&tiles, rows, columns)
}

/// The temporal clips a long video is decoded in, as a family's frame rule states them: each
/// clip decodes a window of latent frames, keeps a body, and leaves an overlap tail its
/// successor fades its first frames against.
///
/// A timeline of `n` clips is `body · n + overlap` frames. Clip `k` reads `window()` latent
/// frames from `k · advance`, decodes them to `window() · temporal` frames of which the leading
/// surplus over `native()` is cropped, keeps its first `body` frames, skips `padding`, and its
/// last `overlap` frames are the tail; the last clip keeps its tail.
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
    pub const fn native(&self) -> u32 {
        self.body + self.padding + self.overlap
    }

    /// The latent frames a clip's window reads: `ceil(native / temporal)`.
    pub fn window(&self) -> u32 {
        self.native().div_ceil(self.temporal.max(1))
    }

    /// The first latent frame of clip `clip`'s window.
    pub const fn start(&self, clip: u32) -> u32 {
        clip * self.advance
    }

    /// The frames of a clip's decode its tail occupies.
    pub const fn tail(&self) -> Range<u32> {
        self.body + self.padding..self.native()
    }

    /// The splices a timeline of `count` clips is written in: clip `i` writes from frame
    /// `body · i`, every clip but the first fades `overlap` frames against its predecessor, and
    /// the last keeps its tail.
    ///
    /// # Errors
    ///
    /// A timeline of no clip.
    pub fn splices(&self, count: u32) -> Result<Vec<Splice>> {
        if count == 0 {
            return Err(Error::shape(
                "TemporalClips",
                "a timeline holds at least one clip".to_owned(),
            ));
        }

        Ok((0..count)
            .map(|index| Splice {
                keep: self.body + if index + 1 == count { self.overlap } else { 0 },
                overlap: if index == 0 { 0 } else { self.overlap },
                first: index * self.body,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one tile of an axis that a single tile covers.
    fn whole(length: usize) -> Vec<Range<usize>> {
        std::iter::once(0..length).collect()
    }

    #[test]
    fn a_raster_within_one_tile_is_one_tile_with_no_seam() {
        assert_eq!(split_tiles(256, 512, 64, 16).unwrap(), whole(256));
        assert_eq!(split_tiles(512, 512, 64, 16).unwrap(), whole(512));

        assert!(split_tiles(0, 512, 64, 16).is_err());
        assert!(split_tiles(1024, 512, 512, 16).is_err(), "no tile advances");
        assert!(
            split_tiles(1000, 512, 64, 16).is_err(),
            "an unaligned extent"
        );
    }

    #[test]
    fn the_slack_spreads_over_the_first_seams_in_alignment_steps() {
        // Three tiles of 512 over 1280 at a minimum overlap of 64: 128 samples of slack, four
        // alignment steps of 32, two to each seam.
        let tiles = split_tiles(1280, 512, 64, 32).unwrap();
        assert_eq!(tiles, [0..512, 384..896, 768..1280]);

        // One step more than the seams divide goes to the first seam.
        let tiles = split_tiles(1312, 512, 64, 32).unwrap();
        assert_eq!(tiles, [0..512, 384..896, 800..1312]);
        assert_eq!(tiles[0].end - tiles[1].start, 128);
        assert_eq!(tiles[1].end - tiles[2].start, 96);
    }

    #[test]
    fn a_timeline_is_body_clips_and_one_kept_tail() {
        let clips = TemporalClips {
            body: 96,
            padding: 9,
            overlap: 16,
            temporal: 4,
            advance: 24,
        };
        assert_eq!(clips.native(), 121);
        assert_eq!(clips.window(), 31);
        assert_eq!(clips.start(2), 48);
        assert_eq!(clips.tail(), 105..121);

        let splices = clips.splices(3).unwrap();
        assert_eq!(
            splices,
            [
                Splice {
                    keep: 96,
                    overlap: 0,
                    first: 0
                },
                Splice {
                    keep: 96,
                    overlap: 16,
                    first: 96
                },
                Splice {
                    keep: 112,
                    overlap: 16,
                    first: 192
                },
            ]
        );
        assert!(clips.splices(0).is_err());
    }
}

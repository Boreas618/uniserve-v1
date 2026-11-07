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

use crate::encoding::{EncoderScratch, Extent, Grid, SPATIAL_RATIO, VideoEncoder};
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
    /// The pixel extent each tile of the run encodes, one per tile in plan order.
    ///
    /// This is what a driver passes to `VideoEncoder::reserve` before its loop: the reservation is
    /// made once for the whole run, because the arena never rewinds inside a forward. A grid's tiles
    /// are uniform - `Seams::split` keeps every tile at the raster's tile size and spends the slack on
    /// wider seams, so the grid's multiset of extents is one - and a raster smaller than a tile is one
    /// tile of the raster's own size.
    pub fn tile_extents(&self, frames: u32) -> Vec<Extent> {
        self.tiles
            .iter()
            .map(|tile| Extent {
                frames,
                height: tile.rows.1 * SPATIAL_RATIO,
                width: tile.columns.1 * SPATIAL_RATIO,
            })
            .collect()
    }

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
    /// Tile `i`'s pixels, in plan order.
    ///
    /// Without [`Self::packed`] each view is `[1, in_channels, frames, tile_height, tile_width]` and
    /// **packed**: `nn::layers::FramePad` pads a frame's rows in place of the source's own, so it
    /// requires a frame's rows to be contiguous in the view it reads. A caller that uploads tiles as
    /// separate buffers passes them as they are. With it, each view is the tile's block of a wider
    /// raster, `[in_channels, frames, tile_height, tile_width]` at the raster's strides.
    pub pixels: &'a [TensorView],
    /// The one buffer a tile cut from a wider raster is packed into, just before it is encoded.
    ///
    /// A tile cut from a raster has the raster's row stride instead of the tile's width, so its
    /// block is copied out first (one tile is a few megabytes against the encoder's set). One buffer
    /// serves every tile because a tile is copied only once the tile before it was encoded.
    pub packed: Option<TensorView>,
    /// The run's stitched latents, `[channels, frames, run_rows, latent_columns]`.
    pub blended: TensorView,
}

/// What the clips of one raster encode in: the encoder's set, every tile's block and the hand-off.
///
/// Reserved once ([`ClipEncoder::reserve`]) and used by each clip of the raster in turn.
pub struct ClipRun {
    scratch: EncoderScratch,
    /// This rank's share of the tiles' blocks.
    own: TensorView,
    /// Every rank's share, in plan order; `own` itself on a rank with no group.
    gathered: TensorView,
    /// The block a two-way fade hands to its second blend.
    handed: TensorView,
    rank: usize,
    ranks: usize,
    /// The tiles of a rank's share.
    share: usize,
}

/// The stitcher's blocks and hand-off: the buffers `Scene::encode` takes after the encoder's set,
/// on one rank.
///
/// Every tile's decoded block, in plan order, and one more block that a tile whose fade is both ways
/// hands to its second blend. The blocks are held until the stitch because a group's tiles are
/// encoded on all of its ranks and reach each of them at once. A block is the caller's raster's
/// channel and frame extent by the tile's latent extent, because the stitcher blends the raster the
/// caller funds and not a shape of its own; `latent_frames` is therefore the raster's frame extent,
/// not the request's. Under a group of `ranks` the tiles are this plan's rounded up to a whole share
/// each, and a rank also holds its own share before the gather.
///
/// A run sized from the encoder's own set alone stops at the first of these buffers, which is why
/// they are stated by name: a plan is what a caller allocates, and bytes it does not name are bytes
/// the forward takes past its arena.
pub fn stitch_arena(
    config: &crate::VideoVaeConfig,
    tiles: usize,
    latent_frames: u64,
    tile_rows: u64,
    tile_columns: u64,
) -> Result<Vec<crate::decoding::Buffer>> {
    // The raster's channel extent, `2 * latent_channels`: the encoder writes a mean and a deviation
    // per latent channel and the clip's plan splits them.
    let block = 2 * u64::from(config.latent_channels) * latent_frames * tile_rows * tile_columns;
    Ok(vec![
        crate::decoding::Buffer {
            name: "stitch.tiles".to_owned(),
            element: "f32",
            elements: block * tiles as u64,
        },
        crate::decoding::Buffer {
            name: "stitch.handed".to_owned(),
            element: "f32",
            elements: block,
        },
    ])
}

/// Copies a tile's block of a wider raster into `buffer` and answers it as the encoder reads it.
fn pack_tile(f: &Forward, block: TensorView, buffer: TensorView) -> Result<TensorView> {
    let (channels, frames, height, width) = (
        block.shape.dim(0),
        block.shape.dim(1),
        block.shape.dim(2),
        block.shape.dim(3),
    );
    let packed = [frames * height * width, height * width, width, 1];
    let mut destination_strides = [0_i64; ops::copy_block::RANK];
    for (axis, stride) in packed.iter().enumerate() {
        destination_strides[axis] = i64::try_from(*stride).map_err(|_| {
            Error::config("a condition tile is larger than i64 strides address".to_owned())
        })?;
    }
    let destination = TensorView::with_strides(
        buffer.offset_elements(0),
        DType::F32,
        Shape::new(&[channels, frames, height, width]),
        Strides::new(&packed),
    );
    nn::layers::CopyBlock::new(DType::F32)?.forward(f, block, destination, &destination_strides)?;
    // The encoder's tile view is the packed copy at the shape it pads from, with the batch axis the
    // scene's contract asks for.
    Ok(TensorView::with_strides(
        buffer.offset_elements(0),
        DType::F32,
        Shape::new(&[1, channels, frames, height, width]),
        Strides::new(&[
            channels * frames * height * width,
            frames * height * width,
            height * width,
            width,
            1,
        ]),
    ))
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

    /// The programs one tiled encoding of `rasters` launches.
    ///
    /// The walk states the whole-frame encoder through [`VideoEncoder::programs`]; a condition's
    /// pixels go through this path instead, because a raster larger than one tile becomes its latent
    /// by tiles that cross-fade where they meet. Two families are this path's own and no other
    /// enumeration reaches them: **the seam blends**, one program per `(extent, axis)` the grid
    /// states, and **the encoder's rows at each tile's extent** rather than at the raster's, which
    /// is where its residual adds and pads run at tile widths.
    ///
    /// `frames` is the piece's own (one for a still); the encoder's rows are stated for both a still
    /// and a generated clip by [`VideoEncoder::programs`], so only the tiles' extents are taken here.
    pub fn programs(
        config: &crate::VideoVaeConfig,
        rasters: &[(u32, u32)],
        frames: u32,
    ) -> Result<Vec<(String, String)>> {
        use crate::programs::Programs;
        use ops::cross_fade::CrossFade as CrossFadeOp;
        let mut programs = Programs::default();
        let mut extents: Vec<(u32, u32)> = Vec::new();
        for (height, width) in rasters {
            let encoder = Self::new(*height, *width)?;
            // The pass-through blend `encode` places a run's first tile through: one position on the
            // rows axis, given the same block twice, so it is exact. It is a launch of its own and no
            // seam width states it.
            let place = CrossFade::new(DType::F32, 1, 2).map_err(Error::from)?;
            programs.launch::<CrossFadeOp>(&place.descriptor());
            for (extent, axis) in encoder.blend_extents() {
                let blend = CrossFade::new(DType::F32, extent, axis).map_err(Error::from)?;
                programs.launch::<CrossFadeOp>(&blend.descriptor());
            }
            // A grid's tiles are uniform and a raster inside one tile is one tile of its own size, so
            // the extents below are that raster's single tile class.
            for extent in encoder.plan().tile_extents(frames) {
                let tile = (extent.height, extent.width);
                if !extents.contains(&tile) {
                    extents.push(tile);
                }
            }
        }
        // The tiles' own extents, each stated as a raster of that size: the encoder's frame
        // operators key on the extent they reduce over, and a tile is a smaller raster, not a
        // different operator.
        for (height, width) in extents {
            programs.extend(crate::encoding::VideoEncoder::programs(
                config,
                &[(height, width)],
            )?);
        }
        Ok(programs.finish())
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
        let frames = scene
            .pixels
            .first()
            .map(|tile| {
                let axis = if scene.packed.is_some() { 1 } else { 2 };
                u32::try_from(tile.shape.dim(axis)).unwrap_or(0)
            })
            .unwrap_or(0);
        let run = self.reserve(f, encoder, plan, &[frames], scene.blended)?;
        self.encode_in(f, encoder, plan, scene, &run)
    }

    /// One reservation for clips of `frames` frames each, whose blocks are at most `raster`'s.
    ///
    /// The arena never rewinds inside a forward, so a reservation inside a loop is multiplied by the
    /// tile count, and one tile's buffers are most of a gigabyte at the served canvas. The grid's
    /// tiles are uniform (`Seams::split` keeps every tile at the raster's tile size and spends the
    /// slack on wider seams), so the plan's tiles are one extent per clip length; the list is passed
    /// whole because a raster smaller than a tile is the case where they would not be. A reference
    /// video's clips run in this one reservation.
    pub fn reserve(
        &self,
        f: &Forward,
        encoder: &VideoEncoder,
        plan: &Plan,
        frames: &[u32],
        raster: TensorView,
    ) -> Result<ClipRun> {
        let extents: Vec<Extent> = frames
            .iter()
            .flat_map(|frames| plan.tile_extents(*frames))
            .collect();
        let scratch = encoder.reserve(f, &extents)?;
        let tile_rows = plan.tiles.first().map(|tile| tile.rows.1).unwrap_or(0);
        let tile_columns = plan.tiles.first().map(|tile| tile.columns.1).unwrap_or(0);
        // **A raster's tiles are shared over the sequence group.** Each rank of a group would
        // otherwise encode every tile of a condition it needs, four times the work and most of a
        // request's first step: a tile is independent of every other, and only the stitch reads two
        // of them. So a rank encodes its share into its own blocks, the blocks are gathered, and
        // every rank stitches all of them, which is what its rows of the prefix read.
        let (rank, ranks) = f.group.map_or((0, 1), |group| {
            (group.rank() as usize, group.ranks() as usize)
        });
        let share = plan.tiles.len().div_ceil(ranks);
        let block = Shape::new(&[
            raster.shape.dim(0),
            raster.shape.dim(1),
            u64::from(tile_rows),
            u64::from(tile_columns),
        ]);
        let stack = |tiles: usize| {
            Shape::new(&[
                tiles as u64,
                block.dim(0),
                block.dim(1),
                block.dim(2),
                block.dim(3),
            ])
        };
        let own = f.arena().tensor(DType::F32, stack(share))?;
        let gathered = match ranks {
            1 => own,
            _ => f.arena().tensor(DType::F32, stack(share * ranks))?,
        };
        Ok(ClipRun {
            scratch,
            own,
            gathered,
            // The block a two-way fade hands to its second blend.
            handed: f.arena().tensor(DType::F32, block)?,
            rank,
            ranks,
            share,
        })
    }

    /// [`Self::encode`] in a reservation of [`Self::reserve`]: one clip's tiles into its raster.
    pub fn encode_in(
        &self,
        f: &Forward,
        encoder: &VideoEncoder,
        plan: &Plan,
        scene: Scene<'_>,
        run: &ClipRun,
    ) -> Result<()> {
        let Scene {
            pixels,
            packed,
            blended,
        } = scene;
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

        let ClipRun {
            scratch,
            own,
            gathered,
            handed,
            rank,
            ranks,
            share,
        } = run;
        let (own, gathered, handed) = (*own, *gathered, *handed);
        let (rank, ranks, share) = (*rank, *ranks, *share);

        let block = Shape::new(&[
            blended.shape.dim(0),
            blended.shape.dim(1),
            u64::from(tile_rows),
            u64::from(tile_columns),
        ]);
        // A clip's blocks lie one after another in the reservation, at this clip's own size: the
        // last clip of a reference video is shorter than the rest.
        let elements = block.numel();
        let block_of = |blocks: TensorView, index: usize| {
            blocks
                .at(blocks.offset_elements(index as u64 * elements))
                .reshape(block)
        };
        let handed = handed.reshape(block);

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
        }
        let mine = (rank * share).min(plan.tiles.len())..((rank + 1) * share).min(plan.tiles.len());
        for index in mine.clone() {
            let decoded = block_of(own, index - mine.start);
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
            let tile_pixels = match packed {
                Some(buffer) => pack_tile(f, pixels[index], buffer)?,
                None => pixels[index],
            };
            encoder.encode_tile(f, scratch, tile_pixels, out)?;
        }
        if ranks > 1 {
            // This clip's blocks alone: a member's share of them lies at the head of its own.
            let shared = share as u64 * elements;
            let sent = own.reshape(Shape::new(&[shared]));
            nn::layers::UlyssesExchange::new(ranks as u64)?.gather(
                f,
                sent,
                gathered.reshape(Shape::new(&[shared * ranks as u64])),
            )?;
        }

        // The tiles are row-major (`Plan::tiles`), so the neighbour above is one row of the grid
        // back and the one to the left the tile before.
        let columns = self.grid.columns.len();
        for (index, tile) in plan.tiles.iter().enumerate() {
            let decoded = block_of(gathered, index);
            let placed = self.place_in(blended, plan, tile, block)?;
            let above = match tile.fades_above() {
                true => Some(block_of(gathered, index - columns)),
                false => None,
            };
            let left = match tile.fades_left() {
                true => Some(block_of(gathered, index - 1)),
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
    use crate::config::CLIP_FRAMES;

    /// The stitcher's plan is every tile's block and a hand-off block, by name.
    ///
    /// `Scene::encode` reserves exactly these, in this order, after the encoder's own set: two rows of
    /// a two-row ring, one block per tile column of the grid, then the block a two-way fade hands to
    /// its second blend. Each block is the caller's raster extent - its channels and its frames - by
    /// the tile's latent extent, which is why the plan takes the frames as an argument rather than
    /// reading them from the request: the stitcher blends the raster the caller funds.
    #[test]
    fn the_stitch_plan_is_every_tile_and_one_handed_block() {
        let config = crate::video_vae::tests::video_config();
        let plan = stitch_arena(&config, 3, 2, 16, 16).expect("the stitcher's plan");
        let block = 2 * u64::from(config.latent_channels) * 2 * 16 * 16;
        let names: Vec<&str> = plan.iter().map(|buffer| buffer.name.as_str()).collect();
        assert_eq!(names, ["stitch.tiles", "stitch.handed"]);
        assert_eq!(plan[0].elements, 3 * block);
        assert_eq!(plan[1].elements, block);
        for buffer in &plan {
            assert_eq!(buffer.element, "f32");
        }

        // A wider raster's grid is more blocks, not bigger ones.
        let wider = stitch_arena(&config, 7, 2, 16, 16).expect("the wider plan");
        assert_eq!(wider[0].elements, 7 * block);
    }

    /// The 768 by 1344 canvas the fixtures use: 48 by 84 latents, four tile rows and seven columns.
    fn canvas() -> ClipEncoder {
        ClipEncoder::new(768, 1344).expect("the verified canvas")
    }

    #[test]
    fn a_raster_within_one_tile_is_one_tile_with_no_seam() {
        let small = ClipEncoder::new(256, 256).expect("one tile");
        let latent = small.latent_extent();
        // A side of one tile has one latent side: the raster over the VAE's spatial compression,
        // and the whole of it is the tile's own.
        assert_eq!(latent, (256 / SPATIAL_RATIO, 256 / SPATIAL_RATIO));
        let plan = small.plan();
        assert_eq!(plan.tiles.len(), 1);
        let tile = plan.tiles[0];
        assert_eq!(tile.rows, (0, latent.0));
        assert_eq!(tile.columns, (0, latent.1));
        assert!(!tile.fades_above() && !tile.fades_left());
        assert_eq!(plan.crop, (0, latent.0));
    }

    #[test]
    fn a_raster_needs_one_compiled_blend_for_each_seam_width_it_has() {
        // The source of truth is the plan's own seams: every fade the run performs is one of the
        // plan's `(extent, axis)` pairs, and the encoder must have compiled exactly those - not one
        // program per seam, and not a pair no seam needs.
        let encoder = canvas();
        let plan = encoder.plan();
        let mut needed: Vec<(u32, u32)> = Vec::new();
        for tile in &plan.tiles {
            if tile.fades_above() {
                needed.push((tile.above, 2));
            }
            if tile.fades_left() {
                needed.push((tile.left, 3));
            }
        }
        needed.sort_unstable();
        needed.dedup();

        let compiled: Vec<(u32, u32)> = encoder
            .blends
            .iter()
            .map(|(extent, axis, _)| (*extent, *axis))
            .collect();
        assert_eq!(compiled, encoder.blend_extents());
        assert_eq!(compiled, needed);

        // The band's two axes have different seam widths, so the pairs are more than one per axis:
        // without that, this test could not tell a row's blend from a column's.
        assert!(needed.iter().any(|(_, axis)| *axis == 2));
        assert!(needed.iter().any(|(_, axis)| *axis == 3));
        assert!(
            needed.iter().map(|(extent, _)| *extent).max()
                != needed.iter().map(|(extent, _)| *extent).min()
        );

        // A raster inside one tile has no seam at all, so it needs and compiles none.
        let small = ClipEncoder::new(256, 256).expect("one tile");
        assert!(small.blend_extents().is_empty());
        assert!(small.blends.is_empty());
    }

    #[test]
    fn a_raster_that_is_not_whole_patches_is_refused_rather_than_truncated() {
        // The encoder reads whole latent patches, so a side that is not a multiple of the spatial
        // compression has no latent extent, and nothing may round it down silently.
        assert!(ClipEncoder::new(256 + SPATIAL_RATIO / 2, 256).is_err());
        assert!(ClipEncoder::new(256, 256 + SPATIAL_RATIO / 2).is_err());
        assert!(ClipEncoder::new(256, 256).is_ok());
        // A served canvas passes: 768 by 1344 and 480 by 832 are both multiples of the compression.
        assert!(ClipEncoder::new(768, 1344).is_ok());
        assert!(ClipEncoder::new(480, 832).is_ok());
    }

    #[test]
    fn a_run_reserves_one_set_of_buffers_because_its_tiles_are_uniform() {
        // The driver reserves the encoder's buffers once, before its loop over the plan's tiles: the
        // arena never rewinds inside a forward, so a per-tile reservation is multiplied by the tile
        // count. That reservation is sound exactly while the plan's tiles share an extent, and
        // `Seams::split` is what guarantees it - it keeps every tile at the raster's tile size and
        // spends the slack of a canvas that does not divide evenly on wider seams, which is why a
        // 1344-pixel raster is seven 256-pixel tiles whose first seams are 80 pixels.
        let encoder = canvas();
        let plan = encoder.plan();
        let extents = plan.tile_extents(CLIP_FRAMES);
        assert_eq!(extents.len(), plan.tiles.len());
        for extent in &extents {
            assert_eq!(*extent, extents[0]);
        }

        // The one extent is the raster's tile, in pixels, and it is the tile's latent extent scaled
        // back up - the same factor the split is aligned to.
        let tile = &plan.tiles[0];
        assert_eq!(
            (extents[0].height, extents[0].width),
            (tile.rows.1 * SPATIAL_RATIO, tile.columns.1 * SPATIAL_RATIO)
        );
        assert_eq!(extents[0].frames, CLIP_FRAMES);

        // A raster smaller than a tile is one tile of the raster's own size, so the same reservation
        // rule holds there: one extent, not the tile's.
        let small = ClipEncoder::new(256, 256).expect("a fourth of a tile");
        let plan = small.plan();
        let extents = plan.tile_extents(CLIP_FRAMES);
        assert_eq!(plan.tiles.len(), 1);
        assert_eq!((extents[0].height, extents[0].width), (256, 256));
    }

    #[test]
    fn the_canvas_plan_covers_every_latent_position() {
        let encoder = canvas();
        let latent = encoder.latent_extent();
        // The latent raster is the canvas over the spatial compression: 768 and 1344 pixels are 48
        // and 84 latent positions.
        assert_eq!(latent, (768 / SPATIAL_RATIO, 1344 / SPATIAL_RATIO));
        let grid = encoder.grid();
        let plan = encoder.plan();
        // One tile per pair of seams, and a run that starts at the raster's own first position.
        assert_eq!(
            plan.tiles.len(),
            grid.rows.starts.len() * grid.columns.starts.len()
        );
        assert_eq!(plan.extent(), (0, latent.0));

        // A tile row owns the latent rows up to the next row's start, and the last owns the rest:
        // every tile starts where its own owner starts, the first owner is position zero and the
        // last owner is the raster's end.
        let owned = grid.owned(latent.0);
        assert_eq!(owned[0], 0);
        assert_eq!(*owned.last().expect("a last owner"), latent.0);
        assert!(owned.windows(2).all(|pair| pair[0] < pair[1]));
        for tile in &plan.tiles {
            assert_eq!(tile.rows.0, owned[tile.row]);
        }

        // Every latent row and column is written by some tile; the seams are written twice, by the
        // tile that owns them and by the neighbour that fades into them.
        let mut rows = vec![0u32; latent.0 as usize];
        let mut columns = vec![0u32; latent.1 as usize];
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
        let grid = encoder.grid();

        // The overlaps are the reference's, in pixels: `split_tiles` on this raster spreads the
        // 64-pixel slack over the seams in tile order and in whole compression steps, which gives
        // three seams of 6, 5 and 5 latents down the 48 rows and six of 5, 5, 5, 5, 4 and 4 across
        // the 84 columns. They are the reference's numbers, not this module's, which is why they
        // are stated: the derivation below reads them through `Grid::blends`.
        assert_eq!(grid.rows.overlaps, vec![96, 80, 80]);
        assert_eq!(grid.columns.overlaps, vec![80, 80, 80, 80, 64, 64]);
        let rows_blend = Grid::blends(&grid.rows);
        let columns_blend = Grid::blends(&grid.columns);

        // The two axes' seams differ, so a tile that fades both ways tells them apart: reading one
        // axis's blend for the other is the defect this assertion exists to catch.
        assert_ne!(rows_blend, columns_blend);

        let plan = encoder.plan();
        for tile in &plan.tiles {
            // A tile row fades over the seam above it, a tile column over the seam to its left; the
            // first of each has nothing before it and fades nothing.
            let above = match tile.row {
                0 => 0,
                row => rows_blend[row - 1],
            };
            let left = match tile.column {
                0 => 0,
                column => columns_blend[column - 1],
            };
            assert_eq!(tile.above, above, "tile row {} fades above", tile.row);
            assert_eq!(tile.left, left, "tile column {} fades left", tile.column);
        }
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
        let latent = encoder.latent_extent().0;
        // The bands worth testing are the ones the geometry makes interesting: a whole tile row, its
        // first row, everything but its first row, and the raster's last row. They follow from the
        // seam positions, so a changed split moves them instead of leaving them stale.
        let owned = encoder.grid().owned(latent);
        let mut bands: Vec<(u32, u32)> = vec![(0, latent)];
        for pair in owned.windows(2) {
            let (start, stop) = (pair[0], pair[1]);
            bands.push((start, stop));
            if start + 1 < stop {
                bands.push((start, start + 1));
                bands.push((start + 1, stop));
            }
        }
        bands.push((latent - 1, latent));
        for (start, stop) in bands {
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
        let latent = encoder.latent_extent().0;
        // Splitting the raster at every tile row's own boundary must tile it: each band's crop is the
        // range asked for, nothing is read before it, and the bands reach the raster's end.
        let mut next = 0;
        for owned in encoder.grid().owned(latent).iter().skip(1) {
            let stop = *owned;
            let band = encoder.band(next, stop).expect("a band");
            assert_eq!(band.crop, (next, stop));
            assert!(band.origin <= next);
            assert!(band.extent().1 >= stop);
            next = stop;
        }
        assert_eq!(next, latent);
        // An empty run and one past the raster are refused rather than clamped.
        assert!(encoder.band(latent, latent).is_err());
        assert!(encoder.band(0, latent + 1).is_err());
    }
}

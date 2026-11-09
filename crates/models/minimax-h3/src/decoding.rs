//! The H3 decoders' plan: how a video is spliced into clips, what a clip of decoded frames keeps, and
//! the layout each decoder hands the media stage.
//!
//! The video VAE decodes one clip at a time and [`crate::output::VideoPostprocessor`] splices the
//! clips into a timeline; the audio VAE decodes one channel-major stereo latent timeline. This
//! module owns that arithmetic and the output layouts, and nothing that touches a device: the
//! packed latents, the decoded clip frames and the PCM are borrowed views the worker hands in and
//! out.
//!
//! ## The timeline
//!
//! A generated video has `17n + 5` frames (`docs/minimax_h3/minimax_h3.md`). Clip
//! `k` reads latent frames `[5k, 5k + 7)`: five that advance the timeline and two that the next clip
//! reads again. Each clip decodes to 25 frames — a 17-frame body, three VAE padding frames, and a
//! five-frame tail — so a clip keeps `[0, 17)` and cross-fades `[20, 25)` into its successor; the last
//! clip's tail ends the video.
//!
//! ## What each decoder hands the media stage
//!
//! [`crate::output`] states the two rasters and their quantization — the video's `[frames, height,
//! width, 3]` bytes and the audio's `[samples, 2]` integers — because they are the media stage's
//! contract rather than this plan's. What belongs here is the arithmetic that decides *which* frames
//! and *which* samples go where: a clip's body and overlap, the halo a decoded audio window carries.
//! The splice itself - drop the padding frames, fade the overlap, write the bytes - is the
//! postprocessor's, because it is the component's second entry point (`video_postprocessor.forward`)
//! and it reads the predecessor's tail, which a single clip's decode never sees.

use crate::config::VideoVaeConfig;
use crate::video_vae::{Canvas, Tile};
use crate::{Error, Result};
use nn::layers::{TileGrid, TileStitch};
use nn::{Forward, Shape, TensorView};

/// The frames one decoded clip holds: a 17-frame body, three padding frames and a five-frame tail.
pub const CLIP_OUTPUT_FRAMES: u32 = 25;

/// Frames of a clip's native output that overlap the next clip.
pub const OVERLAP_FRAMES: u32 = 5;

/// The video decoder's tiled output path: a clip's tokens become its raster in one launch.
///
/// The transformer leaves every tile's tokens in one buffer, and each pixel of the clip's raster is a
/// function of at most three of them (`ops::tile_stitch`), which is the tiled VAE's stitch with the
/// unpack in front of it: a tile's rows and
/// columns fade against the tile above and the tile to its left as those were decoded, and what the
/// next tile overlaps is cropped. The grid reaches the kernel as the layout's `video_stitch` table
/// ([`stitch_table`]), so one module serves every canvas.
pub struct ClipDecoder {
    stitch: TileStitch,
    /// Pixels per latent position, the ratio a tile's latent extent scales by.
    ratio: u32,
    /// Elements of one token: `out_channels · temporal · spatial²`, the row width every tile's decode
    /// writes and the stitch reads.
    patch: u64,
}

impl ClipDecoder {
    /// The output path of `config`'s canvases, in the raster's declared element type.
    pub fn new(config: &VideoVaeConfig) -> Result<Self> {
        let spatial = config.spatial_compression();
        let temporal = config.temporal_compression();
        let stitch = TileStitch::new(
            crate::video_vae::ACTIVATION,
            config.out_channels,
            temporal,
            spatial,
        )?;
        let patch = u64::from(config.out_channels)
            * u64::from(temporal)
            * u64::from(spatial)
            * u64::from(spatial);
        Ok(Self {
            stitch,
            patch,
            ratio: spatial,
        })
    }

    /// The one stitch this decoder launches, for a walk without a device.
    pub fn stitch(&self) -> &TileStitch {
        &self.stitch
    }

    /// Decode one clip: every tile of its canvas, then the stitch into the clip's raster.
    ///
    /// The runner hands this one latent block and one output and nothing else: `latents` is the clip's
    /// `[voxels, latent_channels]` rows in tile order, `positions` the layout's `video_positions`
    /// `[3, rows]` table, `table` its `video_stitch` table, and `out` the clip's raster `[channels,
    /// frames, rows, columns]` in the decoder's element type. The transformer runs over every tile at
    /// once and leaves each tile's tokens in one buffer, which the stitch reads.
    #[expect(
        clippy::too_many_arguments,
        reason = "the clip's latents, its two layout tables and its output are separate device views"
    )]
    pub fn decode_clip(
        &self,
        f: &Forward,
        decoder: &crate::video_vae::VideoDecoder,
        tiles: &[Tile],
        latents: TensorView,
        positions: TensorView,
        table: TensorView,
        out: TensorView,
    ) -> Result<()> {
        let voxels: u64 = tiles.iter().map(|tile| tile.canvas.voxels()).sum();
        let channels = u64::from(crate::output::Rgb::CHANNELS);
        if latents.shape.rank() != 2 || latents.shape.dim(0) != voxels {
            return Err(Error::config(format!(
                "the clip's latent block is [{voxels}, latent_channels], not {:?}",
                latents.shape.dims()
            )));
        }
        // One table serves every tile only when the grid is uniform, which it is wherever the canvas
        // is at least one tile wide: the coordinates are the tile's own extents, so a tile of another
        // size would need a table of its own.
        let table_rows = positions.shape.dim(1);
        if positions.shape.rank() != 2 || positions.shape.dim(0) != 3 {
            return Err(Error::config(format!(
                "the position table is [3, rows], not {:?}",
                positions.shape.dims()
            )));
        }
        for tile in tiles {
            if crate::video_vae::Canvas::rows(&tile.canvas) != table_rows {
                return Err(Error::config(format!(
                    "tile ({}, {}) needs {} position rows and the layout's table holds {table_rows}: \
                     the table is the largest tile's, so a grid of two tile sizes has none",
                    tile.row,
                    tile.column,
                    crate::video_vae::Canvas::rows(&tile.canvas)
                )));
            }
        }
        if out.shape.rank() != 4 || out.shape.dim(0) != channels {
            return Err(Error::config(format!(
                "the clip's raster is [{channels}, frames, rows, columns], not {:?}",
                out.shape.dims()
            )));
        }
        let (height, width) = (out.shape.dim(2), out.shape.dim(3));
        if table.shape.dims() != [height + width, ops::tile_stitch::TABLE_WIDTH] {
            return Err(Error::config(format!(
                "the stitch table is {:?}, and a raster of {height} rows and {width} columns reads \
                 one row per position of each axis",
                table.shape.dims()
            )));
        }
        // **One canvas per decoder object: the tiles handed here are the grid of the canvas the decoder
        // was built for.** The grid the tiles cover must be one the decoder serves, and a canvas
        // outside the set is refused by name.
        let (covered_frames, covered_height, covered_width) =
            tiles
                .iter()
                .fold((0_u32, 0_u32, 0_u32), |(frames, height, width), tile| {
                    // The tile's **latent span**, not its grid index: `row`/`column` count tiles and the
                    // extent a grid covers is the end of each tile's span.
                    (
                        frames.max(tile.canvas.frames),
                        height.max(tile.latent_rows.1),
                        width.max(tile.latent_columns.1),
                    )
                });
        let canvas = decoder
            .canvases()
            .iter()
            .copied()
            .find(|canvas| {
                (canvas.frames, canvas.height, canvas.width)
                    == (covered_frames, covered_height, covered_width)
            })
            .ok_or_else(|| {
                Error::config(format!(
                    "the tiled decode's tiles cover {covered_height}x{covered_width} latent cells \
                     of {covered_frames} frame(s), and this decoder serves {}",
                    decoder
                        .canvases()
                        .iter()
                        .map(|canvas| format!(
                            "{}x{} of {} frame(s)",
                            canvas.height, canvas.width, canvas.frames
                        ))
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })?;
        let first = tiles.first().ok_or_else(|| {
            Error::config("a clip of no tiles has no raster to stitch".to_owned())
        })?;
        let tile_rows = crate::video_vae::Canvas::rows(&first.canvas);
        let tokens = f.arena().tensor(
            out.dtype,
            Shape::new(&[tile_rows * tiles.len() as u64, self.patch]),
        )?;
        let scratch = decoder.scratch(f, &canvas)?;
        decoder.decode(f, &canvas, latents, positions, tokens, &scratch)?;

        let narrow = |value: u64, name: &str| {
            u32::try_from(value)
                .map_err(|_| Error::config(format!("the grid's {name} {value} is too large")))
        };
        let grid = TileGrid {
            columns: narrow(
                tiles.iter().filter(|tile| tile.row == first.row).count() as u64,
                "column count",
            )?,
            token_rows: narrow(tile_rows, "token rows")?,
            height: first.canvas.height * self.ratio,
            width: first.canvas.width * self.ratio,
        };
        let row_table = TensorView::new(
            table.offset_elements(0),
            table.dtype,
            Shape::new(&[height, ops::tile_stitch::TABLE_WIDTH]),
        );
        let column_table = TensorView::new(
            table.offset_elements(height * ops::tile_stitch::TABLE_WIDTH),
            table.dtype,
            Shape::new(&[width, ops::tile_stitch::TABLE_WIDTH]),
        );
        Ok(self
            .stitch
            .forward(f, tokens, row_table, column_table, grid, out)?)
    }
}

/// The rows of one axis of the `video_stitch` table: for each position, the tile that keeps it, the
/// position inside that tile, and the width of the seam the position lies in.
///
/// A tile keeps everything up to where the next one starts, which is the reference's crop: its
/// trailing overlap is dropped and the next tile's fade stands in that place. A position in the first
/// `overlap` of a tile lies in the seam with its predecessor.
pub fn stitch_axis(seams: &crate::video_vae::Seams, length: u32) -> Vec<[i32; 3]> {
    let mut rows = Vec::with_capacity(length as usize);
    let mut tile = 0_usize;
    for position in 0..length {
        while tile + 1 < seams.len() && seams.starts[tile + 1] <= position {
            tile += 1;
        }
        let inside = position - seams.starts[tile];
        let seam = tile
            .checked_sub(1)
            .and_then(|previous| seams.overlap(previous))
            .filter(|overlap| inside < *overlap)
            .unwrap_or(0);
        rows.push([tile as i32, inside as i32, seam as i32]);
    }
    rows
}

/// The `video_stitch` table of `canvas`: its rows' axis, then its columns'.
pub fn stitch_table(config: &VideoVaeConfig, canvas: &Canvas) -> Result<Vec<[i32; 3]>> {
    let tiles = crate::video_vae::Tiles::of(config, canvas)?;
    let (height, width) = canvas.pixels(config);
    let mut table = stitch_axis(&tiles.rows, height);
    table.extend(stitch_axis(&tiles.columns, width));
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video_vae::Tiles;

    #[test]
    fn the_clip_rows_invert_the_packing_over_each_clips_seven_frames() {
        let canvas = crate::config::Canvas {
            short_edge: 768,
            aspect: (7, 4),
            width: 1344,
            height: 768,
        };
        let patch_rows = u64::from(
            crate::packing::latent_raster(&canvas)
                .expect("raster")
                .patch_rows(),
        );
        // Dense packing keeps raster order, so the clip rows are the raster rows in sequence.
        let rows = crate::packing::clip_rows(crate::config::AttentionKind::Dense, 107, &canvas)
            .expect("rows");
        assert_eq!(rows.len() as u64, 6 * 7 * patch_rows);
        // Clips start five latent frames after the previous one and read seven, so clip `k`'s rows are
        // the raster's rows from frame `5k` — not from `7k`, which is what a clip without overlap
        // would give.
        for (index, row) in rows.iter().enumerate() {
            let clip = index as u64 / (7 * patch_rows);
            let within = index as u64 % (7 * patch_rows);
            assert_eq!(u64::from(*row), clip * 5 * patch_rows + within);
        }
        // A sparse packing permutes them within the generated video. Consecutive clips share their
        // last two latent frames, so a row is named once per clip that reads it — the union covers the
        // generated rows, and no row is named twice inside one clip.
        let sparse = crate::config::AttentionKind::Sparse {
            tile: 128,
            sparsity: 0.8,
            reference_keep: None,
        };
        let rows = crate::packing::clip_rows(sparse, 107, &canvas).expect("rows");
        assert_eq!(rows.len() as u64, 6 * 7 * patch_rows);
        let per_frame = patch_rows as usize;
        let clip_rows = 7 * per_frame;
        let mut generated = vec![false; (5 * 6 + 2) as usize * per_frame];
        for (clip, block) in rows.chunks(clip_rows).enumerate() {
            let mut in_clip = vec![false; generated.len()];
            for row in block {
                let index = *row as usize;
                assert!(
                    index < generated.len(),
                    "clip {clip} reads row {index} outside the video"
                );
                assert!(!in_clip[index], "clip {clip} names row {index} twice");
                in_clip[index] = true;
                generated[index] = true;
            }
        }
        assert!(
            generated.iter().all(|row| *row),
            "a generated row is read by no clip"
        );
        // Clips overlap by two latent frames: the second clip starts five frames after the first.
        assert_eq!(rows[7 * per_frame], rows[5 * per_frame]);
    }

    /// The table states the reference's crop and its seams: every position belongs to one tile, that
    /// tile keeps everything up to where the next one starts, and the first `overlap` positions of a
    /// tile fade against its predecessor. The rectangle is the served canvas, whose two axes have
    /// different grids, so a transposed table cannot pass.
    #[test]
    fn the_stitch_table_keeps_each_position_in_one_tile_and_names_its_seam() {
        let config = crate::video_vae::tests::video_config();
        let canvas = Canvas::clip(&config, 48, 84);
        let tiles = Tiles::of(&config, &canvas).expect("the grid");
        let (height, width) = canvas.pixels(&config);
        let table = stitch_table(&config, &canvas).expect("the table");
        assert_eq!(table.len() as u32, height + width);
        for (axis, rows, length) in [
            (&tiles.rows, &table[..height as usize], height),
            (&tiles.columns, &table[height as usize..], width),
        ] {
            for position in 0..length {
                let [tile, inside, seam] = rows[position as usize];
                let tile = tile as usize;
                assert_eq!(axis.starts[tile] + inside as u32, position);
                assert!(inside < crate::video_vae::TILE_PIXELS as i32);
                // The next tile starts exactly where this one's kept span ends.
                if let Some(next) = axis.starts.get(tile + 1) {
                    assert!(position < *next);
                }
                let expected = match tile {
                    0 => 0,
                    _ if (inside as u32) < axis.overlaps[tile - 1] => axis.overlaps[tile - 1],
                    _ => 0,
                };
                assert_eq!(seam as u32, expected, "position {position}");
            }
            // Every tile keeps something, in order.
            let last = rows.last().expect("an axis has a position")[0];
            assert_eq!(last as usize + 1, axis.len());
        }
    }

    /// A one-tile axis has no seam: the whole axis is the tile's.
    #[test]
    fn an_axis_of_one_tile_has_no_seam() {
        let seams = crate::video_vae::Seams::split(128, 256, 64, 16).expect("one tile");
        let rows = stitch_axis(&seams, 128);
        assert!(rows.iter().all(|row| row[0] == 0 && row[2] == 0));
        assert_eq!(rows[127], [0, 127, 0]);
    }
}

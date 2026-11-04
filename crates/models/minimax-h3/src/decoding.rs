//! The H3 decoders' plan: how a video is spliced into clips, what a clip of decoded frames keeps, and
//! the layout each decoder hands the media stage.
//!
//! `uniserve/models/minimax_h3/decoding.py` and `output.py` are the reference. The video VAE decodes
//! one clip at a time and [`crate::output::VideoPostprocessor`] splices the clips into a timeline; the
//! audio VAE decodes one
//! channel-major stereo latent timeline. This module owns that arithmetic and the output layouts, and
//! nothing that touches a device: the packed latents, the decoded clip frames and the PCM are borrowed
//! views the executor hands in and out.
//!
//! ## The timeline
//!
//! A generated video has `17n + 5` frames (specs/worker.md 9, `docs/minimax_h3/minimax_h3.md`). Clip
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
use crate::video_vae::{Canvas, Tile, Tiles};
use crate::{
    Error, Result,
    config::{CLIP_FRAMES, CLIP_LATENTS},
};
use nn::layers::{CrossFade, RgbBytes, UnpatchifyVideo};
use nn::{DType, Forward, Shape, TensorView};

/// The frames one decoded clip holds: a 17-frame body, three padding frames and a five-frame tail.
pub const CLIP_OUTPUT_FRAMES: u32 = 25;

/// Frames of a clip's native output that overlap the next clip.
pub const OVERLAP_FRAMES: u32 = 5;

/// The first frame of a clip's overlap tail, within its own 25 decoded frames.
pub const TAIL_START: u32 = tail_start_of(OVERLAP_FRAMES);

/// The first frame of a tail of `overlap` frames: the clip's native output less the tail.
const fn tail_start_of(overlap: u32) -> u32 {
    CLIP_OUTPUT_FRAMES - overlap
}

/// One clip of a video's timeline.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Clip {
    /// The index of the clip in the timeline.
    pub index: u32,
    /// The clip's latent frames `[start, end)`: seven, of which the last two repeat in the next clip.
    pub latent: (u32, u32),
    /// The frames of the clip's own 25 that the timeline keeps: the 17-frame body.
    pub body: (u32, u32),
    /// The frames of the clip's own 25 that overlap the next clip: the five-frame tail.
    pub tail: (u32, u32),
}

impl Clip {
    /// The weight of this clip across its tail, one per overlap frame.
    ///
    /// The reference blends with `i / 5` for `i = 0..5`, so the first blended frame is entirely the
    /// predecessor's and the fifth entirely this clip's. The weights are computed in the decoded
    /// element type there; this is the same table stated exactly, which a quantized implementation
    /// compares against.
    pub fn overlap_weights(&self) -> [f32; OVERLAP_FRAMES as usize] {
        let mut weights = [0.0; OVERLAP_FRAMES as usize];
        for (index, weight) in weights.iter_mut().enumerate() {
            *weight = index as f32 / OVERLAP_FRAMES as f32;
        }
        weights
    }
}

/// The clips of a `17n + 5`-frame video, in timeline order.
///
/// Every clip covers a 17-frame body; the last clip adds the five-frame tail, so the number of clips
/// is `(frames - 5) / 17`.
pub fn clips(frames: u32) -> Result<Vec<Clip>> {
    if frames < CLIP_FRAMES + CLIP_LATENTS || !(frames - CLIP_LATENTS).is_multiple_of(CLIP_FRAMES) {
        return Err(Error::config(format!(
            "an H3 video of {frames} frames is not {CLIP_FRAMES}n + {CLIP_LATENTS}"
        )));
    }
    let count = (frames - CLIP_LATENTS) / CLIP_FRAMES;
    Ok((0..count)
        .map(|index| Clip {
            index,
            latent: (
                index * CLIP_LATENTS,
                index * CLIP_LATENTS + CLIP_LATENTS + 2,
            ),
            body: (0, CLIP_FRAMES),
            tail: (TAIL_START, TAIL_START + CLIP_LATENTS),
        })
        .collect())
}

/// The audio timeline of one request: the samples it decodes to and the latent frames it needs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Track {
    /// Interleaved sample frames the decoder writes.
    pub samples: u32,
    /// Latent frames the decoder reads, one per 800 samples.
    pub latent_frames: u32,
    /// Latent frames of context each side of a decoded window depends on.
    pub halo: u32,
}

impl Track {
    /// The track of a video's `frames` at the checkpoint's 40 Hz latent rate.
    pub fn of(frames: u32, audio: &crate::AudioVaeConfig) -> Result<Self> {
        let samples = crate::packing::audio_samples(frames);
        Ok(Self {
            samples,
            latent_frames: crate::packing::audio_latent_frames(frames),
            halo: crate::audio_vae::receptive_field(audio)?,
        })
    }

    /// The latent frames a decoded window `[start, stop)` needs, halo included and clipped to the
    /// track.
    pub fn window(&self, start: u32, stop: u32) -> (u32, u32) {
        (
            start.saturating_sub(self.halo),
            stop.saturating_add(self.halo).min(self.latent_frames),
        )
    }
}

/// One scratch buffer of a decoder's forward, with the element type it holds.
///
/// The executor sizes the arena from these; the model never allocates. Names are the ones the arena
/// plan in `notes/h3/decoders-interface.md` uses, so a mismatch is visible in one place.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Buffer {
    /// The buffer's name.
    pub name: &'static str,
    /// Its element type: `f16`, `f32` or `u8`.
    pub element: &'static str,
    /// How many elements it holds.
    pub elements: u64,
}

impl Buffer {
    /// The bytes the buffer occupies.
    pub fn bytes(&self) -> u64 {
        let width = match self.element {
            "u8" => 1,
            "f16" => 2,
            _ => 4,
        };
        self.elements * width
    }
}

/// The clip-latent row of every voxel of every tile, in tile order then voxel order, with the number of
/// voxels each tile contributes.
///
/// A tile is a rectangle of the clip's latent raster — every frame, a run of rows and a run of columns —
/// so its voxels are a subset of the clip's, and a tile's rows cannot be a strided view of the clip's
/// when the tile is narrower than the raster. This table names them: the runner gathers each tile's
/// `[voxels_tile, latent_channels]` block out of the clip's one `[frames·h·w, latent_channels]` latent
/// with `nn::layers::Rows`, which is model layout arithmetic, uploaded once per layout.
///
/// `tiles` are in grid order ([`Tiles::tiles`]), and each tile contributes
/// `frames · (rows_end − rows_start) · (columns_end − columns_start)` entries: the row index
/// `((frame · h + row) · w + column)` of each of its voxels.
///
/// The counts are returned with the rows rather than left implicit, because they are not derivable from
/// the tile size alone: a canvas side shorter than one tile gives a single tile of that side
/// (`Seams::split`'s one-tile branch), so a block can be smaller than the tile. Where the side is at
/// least a tile, every tile is exactly the tile's extent — `Seams::split` builds `lengths = [tile; n]`
/// and distributes the slack over the seams — so the grid is uniform and the counts are all equal.
///
/// **The tiles overlap at their seams**, so the entries are not a permutation of the clip's voxels: a
/// seam's positions are decoded by both neighbours, which is what the blend is for.
pub fn tile_rows(canvas: &Canvas, tiles: &[Tile]) -> (Vec<i32>, Vec<u32>) {
    let height = u64::from(canvas.height);
    let width = u64::from(canvas.width);
    let mut rows = Vec::with_capacity(canvas.voxels() as usize);
    let mut counts = Vec::with_capacity(tiles.len());
    for tile in tiles {
        let (first_row, last_row) = tile.latent_rows;
        let (first_column, last_column) = tile.latent_columns;
        let before = rows.len();
        for frame in 0..u64::from(canvas.frames) {
            for row in u64::from(first_row)..u64::from(last_row) {
                for column in u64::from(first_column)..u64::from(last_column) {
                    rows.push(((frame * height + row) * width + column) as i32);
                }
            }
        }
        counts.push((rows.len() - before) as u32);
    }
    (rows, counts)
}

/// The scratch one video clip's decode needs, one entry per distinct lifetime.
///
/// A clip decodes every tile of its grid in one call (`TiledDecoder.decode` batches them), so the
/// token buffers hold all tiles at once: [`crate::video_vae::Tile`]s are all the same shape and the
/// stitched crop trims the last to the raster. The buffers are the reference's own: a residual stream
/// and its normalization, the fused query-key-value projection with its rotary factors, the attention
/// output, the gated feed-forward, the decoded pixels of one tile batch, and the RGB bytes with the
/// overlap block carried between clips.
pub fn video_arena(
    config: &crate::VideoVaeConfig,
    canvas: &crate::video_vae::Canvas,
) -> Result<Vec<Buffer>> {
    let tiles = crate::video_vae::Tiles::of(config, canvas)?;
    let grid = tiles.tiles(config, canvas);
    let ffn = u64::from(config.decoder_ffn_mult);
    let (rotated, _) = crate::video_vae::rotary(config)?;
    let columns = u64::from(config.decoder_num_attention_heads)
        * u64::from(config.decoder_attention_head_dim);
    let (height, width_px) = canvas.pixels(config);
    let frames = canvas.frames_out(config);
    // One token buffer holds every tile, so its rows are the tiles' rows in grid order — summed,
    // because a tile at the canvas's edge can be narrower than the tile size.
    let rows: u64 = grid.iter().map(|tile| tile.canvas.rows()).sum();
    let voxels: u64 = grid.iter().map(|tile| tile.canvas.voxels()).sum();
    // A tile's decoded block is its own latent extent in output pixels.
    let patches: u64 = grid
        .iter()
        .map(|tile| {
            let (tile_height, tile_width) = tile.canvas.pixels(config);
            u64::from(frames) * u64::from(tile_height) * u64::from(tile_width)
        })
        .sum();
    // The clip's own raster, which the tiles are stitched into.
    let raster = u64::from(frames) * u64::from(height) * u64::from(width_px);
    Ok(vec![
        Buffer {
            name: "stream",
            element: "f16",
            elements: rows * columns,
        },
        Buffer {
            name: "projected",
            element: "f16",
            elements: voxels * columns,
        },
        Buffer {
            name: "normalized",
            element: "f16",
            elements: rows * columns,
        },
        Buffer {
            name: "query",
            element: "f16",
            elements: rows * columns,
        },
        Buffer {
            name: "key",
            element: "f16",
            elements: rows * columns,
        },
        Buffer {
            name: "value",
            element: "f16",
            elements: rows * columns,
        },
        Buffer {
            name: "rotated_query",
            element: "f16",
            elements: rows * columns,
        },
        Buffer {
            name: "rotated_key",
            element: "f16",
            elements: rows * columns,
        },
        Buffer {
            name: "attended",
            element: "f16",
            elements: rows * columns,
        },
        Buffer {
            name: "update",
            element: "f16",
            elements: rows * columns,
        },
        Buffer {
            name: "gate_up",
            element: "f16",
            elements: rows * 2 * columns * ffn,
        },
        Buffer {
            name: "gated",
            element: "f16",
            elements: rows * columns * ffn,
        },
        Buffer {
            name: "rotary",
            element: "f32",
            elements: rows * u64::from(rotated),
        },
        Buffer {
            name: "pixels",
            element: "f16",
            elements: patches * u64::from(crate::output::Rgb::CHANNELS),
        },
        Buffer {
            name: "decoded",
            element: "f16",
            elements: raster * u64::from(crate::output::Rgb::CHANNELS),
        },
        Buffer {
            name: "blended",
            element: "f16",
            elements: raster * u64::from(crate::output::Rgb::CHANNELS),
        },
        Buffer {
            name: "rgb",
            element: "u8",
            elements: raster * u64::from(crate::output::Rgb::CHANNELS),
        },
        Buffer {
            name: "overlap",
            element: "f16",
            elements: crate::output::Rgb::of(frames, height, width_px).overlap_elements(),
        },
    ])
}

/// The video decoder's tiled output path: one tile's tokens become its pixels in the clip's raster.
///
/// The decoder decodes overlapping tiles one after another and blends each into the raster as it
/// lands, which is the reference's stitch (`uniserve/nn/vae/tiled.py:46-81`): a tile is written where
/// it belongs, then blended with the tile above it over the rows they share, then with the tile to
/// its left over the columns they share. The first tile row and column have no predecessor, so they
/// are written and nothing is blended.
///
/// Two rasters of `[channels, frames, rows, columns]` in the decoder's element type carry one clip
/// while it is stitched. `plain` holds every tile as it was decoded and is never blended into;
/// `blended` holds the result. The reference keeps its neighbours unmodified for a reason
/// (`uniserve/nn/vae/tiled.py:51-56`): a tile's row blend reads the tile above, and that neighbour
/// must be its *decoded* block rather than one already faded into its own predecessor, or the
/// overlap where two seams cross is computed in a different order from the seams themselves. Writing
/// each tile into both rasters costs one extra unpatchify launch per tile and no copy operator: every
/// placement is a strided view of a raster, which the operators now write through.
pub struct ClipDecoder {
    unpatchify: UnpatchifyVideo,
    /// One blend layer for each overlap the tile grid has, by extent and axis.
    blends: Vec<(u32, u32, CrossFade)>,
    /// Pixels per latent position, the ratio a tile's latent origin scales by.
    ratio: u32,
    /// The raster's frames: a latent frame decodes to `temporal` of them, and every tile spans them
    /// all.
    frames: u32,
    /// The de-normalization into bytes: the ImageNet constants and the rounding, in the decoder's own
    /// element type so the fade's result is quantized without a conversion.
    rgb: RgbBytes,
}

/// The device views one clip's decode reads and writes, as the caller uploaded and reserved them.
///
/// One latent view and one position table per tile of the clip, one token buffer for the whole clip,
/// and the two rasters the tiles are stitched into.
#[derive(Clone, Copy, Debug)]
pub struct Scene<'a> {
    /// Tile `i`'s latent voxel rows, `[voxels, latent_channels]`, in tile order.
    pub latents: &'a [TensorView],
    /// Tile `i`'s rotary table, `[3, rows]` axis-major, in tile order.
    pub positions: &'a [TensorView],
    /// One buffer for the clip, `[tiles · rows, out_channels · temporal · spatial²]`.
    pub tokens: TensorView,
    /// The clip's raster as decoded: read as a neighbour, never written by a blend.
    pub decoded: TensorView,
    /// The clip's raster as stitched: the blends' destination.
    pub blended: TensorView,
}

impl ClipDecoder {
    /// The output path of `canvas`, whose tiles the caller decodes in [`Tile`] order.
    pub fn new(
        config: &VideoVaeConfig,
        canvas: &Canvas,
        mean: nn::Weight,
        std: nn::Weight,
    ) -> Result<Self> {
        let tiles = Tiles::of(config, canvas)?;
        let spatial = config.spatial_compression();
        let unpatchify = UnpatchifyVideo::new(
            DType::F16,
            u64::from(config.out_channels),
            config.temporal_compression(),
            spatial,
        )?;
        // The grid spreads its slack over the seams, so the overlaps are not all equal; one compiled
        // blend serves every seam of the same width, and a decoder fades both ways needs both axes.
        let mut extents: Vec<u32> = tiles
            .rows
            .overlaps
            .iter()
            .chain(tiles.columns.overlaps.iter())
            .copied()
            .filter(|extent| *extent != 0)
            .collect();
        extents.sort_unstable();
        extents.dedup();
        let mut blends = Vec::with_capacity(2 * extents.len());
        for extent in extents {
            for axis in [2, 3] {
                blends.push((extent, axis, CrossFade::new(DType::F16, extent, axis)?));
            }
        }
        let rgb = RgbBytes::new(DType::F16, crate::output::Rgb::CHANNELS, mean, std)?;
        Ok(Self {
            unpatchify,
            blends,
            ratio: spatial,
            frames: canvas.frames_out(config),
            rgb,
        })
    }

    /// The blend layer of `extent` along `axis`.
    fn blend(&self, extent: u32, axis: u32) -> Result<CrossFade> {
        self.blends
            .iter()
            .find(|(width, side, _)| *width == extent && *side == axis)
            .map(|(_, _, layer)| *layer)
            .ok_or_else(|| {
                Error::config(format!(
                    "the clip decoder has no {extent}-pixel blend along axis {axis}"
                ))
            })
    }

    /// Write one decoded tile into the clip's raster and blend it with its neighbours.
    ///
    /// `tokens` is what [`crate::video_vae::VideoDecoder::decode`] wrote for this tile,
    /// `[rows, out_channels · temporal · spatial²]`, its voxel rows first; `dest` is the whole clip's
    /// raster. The caller decodes the tiles in [`Tile`] order, because a blend reads what the tile
    /// above and the tile to the left have already written.
    pub fn write_tile(
        &self,
        f: &Forward,
        tile: &Tile,
        tokens: TensorView,
        decoded: TensorView,
        blended: TensorView,
    ) -> Result<()> {
        for (name, raster) in [("decoded", decoded), ("blended", blended)] {
            let shape = raster.shape;
            if shape.rank() != 4 || shape.dim(0) != u64::from(self.unpatchify.channels()) {
                return Err(Error::config(format!(
                    "the {name} raster is {:?}, not [channels, frames, rows, columns]",
                    shape.dims()
                )));
            }
        }
        let ratio = u64::from(self.ratio);
        let first_row = u64::from(tile.latent_rows.0) * ratio;
        let first_column = u64::from(tile.latent_columns.0) * ratio;
        let plain = self.placement(decoded, tile, first_row, first_column, 0, 0)?;
        let view = self.placement(blended, tile, first_row, first_column, 0, 0)?;
        self.unpatchify.forward(f, tokens, plain)?;
        self.unpatchify.forward(f, tokens, view)?;
        // The blend can only read a neighbour that is one contiguous block. `cross_fade`'s sources are
        // addressed flat in `[channels, frames, rows, columns]` order and only its destination carries
        // strides (its program's own note says so), so a shift of a raster wider than the tile reads
        // the wrong rows past the first. Refusing is the honest answer until the operator reads its
        // sources through their strides: a silently misread seam is a different picture, not a slower
        // one. Single-column grids are unaffected, because there the tile's rows are the raster's.
        if (tile.above != 0 || tile.left != 0) && view.shape.dim(3) != decoded.shape.dim(3) {
            return Err(Error::Unsupported(format!(
                "the clip decoder cannot blend tile ({}, {}) of a {}x{} raster: a neighbour of a \
                 {}-wide tile is not one contiguous block, and cross_fade reads its sources contiguously",
                tile.row,
                tile.column,
                decoded.shape.dim(2),
                decoded.shape.dim(3),
                view.shape.dim(3)
            )));
        }
        if tile.above != 0 {
            let previous = self.placement(decoded, tile, first_row, first_column, tile.above, 0)?;
            self.blend(tile.above, 2)?
                .forward(f, Some(previous), view, view)?;
        }
        if tile.left != 0 {
            let previous = self.placement(decoded, tile, first_row, first_column, 0, tile.left)?;
            self.blend(tile.left, 3)?
                .forward(f, Some(previous), view, view)?;
        }
        Ok(())
    }

    /// A view of `dest` holding one tile's pixel block, shifted back by `up` rows and `left` columns.
    ///
    /// The shift reads the region a blend's predecessor occupies, which the tile grid has already
    /// written; the view keeps the raster's own strides, so the operator writes through them.
    fn placement(
        &self,
        dest: TensorView,
        tile: &Tile,
        first_row: u64,
        first_column: u64,
        up: u32,
        left: u32,
    ) -> Result<TensorView> {
        let ratio = u64::from(self.ratio);
        let rows = first_row
            .checked_sub(u64::from(up))
            .ok_or_else(|| Error::config("a blend starts above the raster".to_owned()))?;
        let columns = first_column
            .checked_sub(u64::from(left))
            .ok_or_else(|| Error::config("a blend starts left of the raster".to_owned()))?;
        let shape = Shape::new(&[
            dest.shape.dim(0),
            u64::from(self.frames),
            u64::from(tile.canvas.height) * ratio,
            u64::from(tile.canvas.width) * ratio,
        ]);
        Ok(TensorView::with_strides(
            dest.element(&[0, 0, rows, columns]),
            dest.dtype,
            shape,
            dest.strides,
        ))
    }

    /// Decode one clip: every tile of its canvas, in [`Tile`] order, into the clip's two rasters.
    ///
    /// `latents[i]` is tile `i`'s latent voxel rows `[voxels, latent_channels]` and `positions[i]` its
    /// `[3, rows]` rotary table, both as the caller uploaded them; `tokens` is one buffer for the whole
    /// clip, `[tiles · rows, out_channels · temporal · spatial²]`, which each tile's decode writes into in
    /// its own row range. `decoded` and `blended` are the clip's rasters: the first holds every tile as it
    /// was decoded and is only ever read as a neighbour, the second carries the stitched result.
    ///
    /// The tiles are the clip's own canvas — [`Tiles::tiles`] — and they are decoded one after another
    /// because a seam blend reads what the tile above and the tile to the left have already written.
    pub fn decode_clip(
        &self,
        f: &Forward,
        decoder: &crate::video_vae::VideoDecoder,
        tiles: &[Tile],
        scene: Scene<'_>,
    ) -> Result<()> {
        let Scene {
            latents,
            positions,
            tokens,
            decoded,
            blended,
        } = scene;
        check_clip(tiles, latents.len(), positions.len(), tokens.shape)?;
        let mut start = 0_u64;
        for (index, tile) in tiles.iter().enumerate() {
            let tile_rows = crate::video_vae::Canvas::rows(&tile.canvas);
            let view = TensorView::new(
                tokens.offset_elements(start * tokens.shape.dim(1)),
                tokens.dtype,
                Shape::new(&[tile_rows, tokens.shape.dim(1)]),
            );
            decoder.decode(f, latents[index], positions[index], view)?;
            self.write_tile(f, tile, view, decoded, blended)?;
            start += tile_rows;
        }
        Ok(())
    }

    /// Quantize one clip's stitched raster into the request's frame range.
    ///
    /// `blended` is the clip's raster — `[channels, frames, rows, columns]` in the decoder's element
    /// type — and `out` the slice of the request's byte raster the clip's frames land in,
    /// `[frames, rows, columns, channels] u8`. The layer de-normalizes with the ImageNet constants,
    /// clamps and rounds, and writes through `out`'s strides, so a clip whose frames are a sub-block of
    /// a larger raster needs no copy.
    pub fn quantize(&self, f: &Forward, blended: TensorView, out: TensorView) -> Result<()> {
        let shape = blended.shape;
        if shape.rank() != 4 || shape.dim(0) != u64::from(crate::output::Rgb::CHANNELS) {
            return Err(Error::config(format!(
                "{:?} is not a [{}, frames, rows, columns] raster",
                shape.dims(),
                crate::output::Rgb::CHANNELS
            )));
        }
        if shape.dim(1) != u64::from(self.frames) {
            return Err(Error::config(format!(
                "the clip decodes {} frames, the raster holds {}",
                self.frames,
                shape.dim(1)
            )));
        }
        self.rgb.forward(f, blended, out)?;
        Ok(())
    }
}

/// Check one clip's arguments before any launch, and answer the token rows it needs.
///
/// A clip decodes one tile per entry of its grid: one latent view and one position table each, and one
/// token buffer whose rows are the tiles' rows in the same order.
fn check_clip(tiles: &[Tile], latents: usize, positions: usize, tokens: Shape) -> Result<u64> {
    if latents != tiles.len() || positions != tiles.len() {
        return Err(Error::config(format!(
            "{} tiles need {} latent views and {} position tables, found {latents} and {positions}",
            tiles.len(),
            tiles.len(),
            tiles.len()
        )));
    }
    let rows: u64 = tiles
        .iter()
        .map(|tile| crate::video_vae::Canvas::rows(&tile.canvas))
        .sum();
    if tokens.rank() != 2 || tokens.dim(0) != rows {
        return Err(Error::config(format!(
            "the clip's token buffer is {:?}, {rows} rows expected",
            tokens.dims()
        )));
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ImageNet constants as the model publishes them: host memory, one f32 per channel.
    fn constants(values: [f32; 3]) -> nn::Weight {
        let mut bytes = Vec::with_capacity(12);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let bytes: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        let ptr = nn::DevicePtr::new(bytes.as_ptr() as u64);
        nn::Weight {
            backing: nn::Backing::new(ptr, bytes.len() as u64),
            view: TensorView::new(ptr, DType::F32, Shape::new(&[3])),
        }
    }

    fn mean() -> nn::Weight {
        constants(crate::output::PIXEL_MEAN)
    }

    fn std() -> nn::Weight {
        constants(crate::output::PIXEL_STD)
    }

    #[test]
    fn the_output_path_refuses_constants_that_are_not_three_channels() {
        let config = crate::video_vae::tests::video_config();
        let canvas = Canvas::clip(&config, 48, 84);
        assert!(ClipDecoder::new(&config, &canvas, mean(), std()).is_ok());
        // Two channels cannot de-normalize an RGB raster; the layer refuses at construction rather than
        // reading past the constant.
        let two = {
            let bytes: &'static [u8] = Box::leak(vec![0u8; 8].into_boxed_slice());
            let ptr = nn::DevicePtr::new(bytes.as_ptr() as u64);
            nn::Weight {
                backing: nn::Backing::new(ptr, bytes.len() as u64),
                view: TensorView::new(ptr, DType::F32, Shape::new(&[2])),
            }
        };
        assert!(ClipDecoder::new(&config, &canvas, two, std()).is_err());
    }

    #[test]
    fn a_four_second_video_is_six_clips() {
        // 4 s at 24 fps rounds to 96 frames, extended to the next 17n + 5: 107.
        assert_eq!(crate::packing::align_num_frames(96), 107);
        let clips = clips(107).expect("six clips");
        assert_eq!(clips.len(), 6);
        assert_eq!(clips[0].latent, (0, 7));
        assert_eq!(clips[5].latent, (25, 32));
        for (index, clip) in clips.iter().enumerate() {
            assert_eq!(clip.index, index as u32);
            assert_eq!(clip.body, (0, 17));
            assert_eq!(clip.tail, (20, 25));
        }
        // The clips cover the timeline exactly: 17 per body plus the last clip's tail.
        let covered = 17 * clips.len() as u32 - 17 + 22;
        assert_eq!(covered, 107);
    }

    #[test]
    fn the_latent_timeline_repeats_two_frames_per_clip() {
        for frames in [107u32, 124, 192, 362] {
            let clips = clips(frames).expect("clips");
            // `5n + 2` latent frames, as packing states.
            let latent = crate::packing::video_latent_frames(frames).expect("latent frames");
            assert_eq!(latent, 5 * clips.len() as u32 - 5 + 7);
            for pair in clips.windows(2) {
                assert_eq!(pair[0].latent.1, pair[1].latent.0 + 2);
            }
        }
    }

    #[test]
    fn a_frame_count_off_the_timeline_is_refused() {
        assert!(clips(96).is_err());
        assert!(clips(100).is_err());
        assert!(clips(0).is_err());
        assert!(clips(22).is_ok() || clips(22).is_err());
        // The smallest legal video: one clip.
        assert_eq!(clips(22).expect("one clip").len(), 1);
    }

    #[test]
    fn the_overlap_weights_are_the_references_fifths() {
        let clip = clips(107).expect("clips")[0];
        assert_eq!(clip.overlap_weights(), [0.0, 0.2, 0.4, 0.6, 0.8]);
        // Every strict increase, and the last below one: the clip never fully replaces its
        // predecessor inside its own tail.
        let weights = clip.overlap_weights();
        for pair in weights.windows(2) {
            assert!(pair[0] < pair[1]);
        }
        assert!(weights[4] < 1.0);
    }

    #[test]
    fn the_audio_track_carries_the_receptive_halo() {
        let config = crate::audio_vae::tests::audio_config();
        let track = Track::of(107, &config).expect("track");
        // packing's audio timeline: 200 latent frames for a five-second track, 800 samples each.
        assert_eq!(track.halo, 30);
        let track = Track::of(124, &config).expect("track");
        assert_eq!(
            track.latent_frames,
            crate::packing::audio_latent_frames(124)
        );
        assert_eq!(track.samples, crate::packing::audio_samples(124));
        // A window inside the track grows by the halo on both sides and is clipped at the ends.
        assert_eq!(track.window(40, 60), (10, 90));
        assert_eq!(track.window(0, 10), (0, 40));
        assert_eq!(
            track.window(track.latent_frames - 5, track.latent_frames),
            (track.latent_frames - 35, track.latent_frames)
        );
    }

    #[test]
    fn the_tile_rows_name_the_voxels_of_every_tile_in_turn() {
        let config = crate::video_vae::tests::video_config();
        let canvas = Canvas::clip(&config, 48, 84);
        let grid = Tiles::of(&config, &canvas)
            .expect("the grid")
            .tiles(&config, &canvas);
        let (rows, counts) = tile_rows(&canvas, &grid);
        // One count per tile, and the counts add up to the table: the table is the tiles' blocks end to
        // end, with no padding between them.
        assert_eq!(counts.len(), grid.len());
        assert_eq!(
            counts.iter().map(|count| u64::from(*count)).sum::<u64>(),
            rows.len() as u64
        );
        // **Tiles overlap in the latent grid**, so the table is longer than the clip's voxels and each
        // seam voxel is named twice: the tile that owns it and the neighbour that re-decodes its band.
        // A tile's block is its own canvas — every frame, every row of its latent rows, every column.
        let voxels = canvas.voxels();
        assert!(
            rows.len() as u64 > voxels,
            "an overlapping grid names more than {} voxels, not fewer",
            voxels
        );
        let mut offset = 0;
        for (tile, count) in grid.iter().zip(&counts) {
            let frame_columns = u64::from(canvas.width);
            let block_rows = u64::from(tile.latent_rows.1 - tile.latent_rows.0);
            let block_columns = u64::from(tile.latent_columns.1 - tile.latent_columns.0);
            assert_eq!(
                u64::from(*count),
                u64::from(canvas.frames) * block_rows * block_columns,
                "tile ({}, {}) names a block of its own canvas",
                tile.row,
                tile.column
            );
            let block = &rows[offset..offset + (*count as usize)];
            // The block is the raster's own row-major order restricted to the tile's rectangle, so it
            // starts at the rectangle's origin and rises all the way through.
            assert_eq!(
                block[0] as u64,
                u64::from(tile.latent_rows.0) * frame_columns + u64::from(tile.latent_columns.0)
            );
            assert!(block.windows(2).all(|pair| pair[0] < pair[1]));
            assert!((*block.last().expect("a block has a voxel") as u64) < voxels);
            offset += *count as usize;
        }
        // The multiplicity is the grid's own geometry, not a bound: a voxel is claimed by exactly the
        // tiles whose latent rectangle contains it. In a two-dimensional grid that is one tile in the
        // interior, two along one seam, and **four** at a corner where a row seam and a column seam
        // cross - so a bound of two is wrong, and the expected count is derived here from the same
        // rectangles the blocks were checked against above.
        let mut coverage = vec![0u8; voxels as usize];
        for row in &rows {
            coverage[*row as usize] += 1;
        }
        // A voxel's index is frame-major, then the latent row, then the column: `frames · height ·
        // width` voxels in all, which is `canvas.voxels()`.
        let plane = u64::from(canvas.height) * u64::from(canvas.width);
        let mut expected = vec![0u8; voxels as usize];
        for tile in grid.iter() {
            for frame in 0..canvas.frames {
                for latent_row in tile.latent_rows.0..tile.latent_rows.1 {
                    for latent_column in tile.latent_columns.0..tile.latent_columns.1 {
                        let voxel = u64::from(frame) * plane
                            + u64::from(latent_row) * u64::from(canvas.width)
                            + u64::from(latent_column);
                        expected[voxel as usize] += 1;
                    }
                }
            }
        }
        assert_eq!(coverage, expected);
        assert!(coverage.iter().all(|count| *count >= 1));
        // The derived count is confirmed by the seam arithmetic rather than by a literal: a voxel can
        // only be claimed by a tile of its row band and a tile of its column band, so the multiplicity
        // is at most the product of the two bands' widths, which is what a corner reaches.
        let mut rows_band = vec![0u32; canvas.height as usize];
        let mut columns_band = vec![0u32; canvas.width as usize];
        for tile in grid.iter() {
            for latent_row in tile.latent_rows.0..tile.latent_rows.1 {
                rows_band[latent_row as usize] += 1;
            }
            for latent_column in tile.latent_columns.0..tile.latent_columns.1 {
                columns_band[latent_column as usize] += 1;
            }
        }
        let bound = u8::try_from(
            rows_band.iter().copied().max().unwrap_or(0) as u64
                * columns_band.iter().copied().max().unwrap_or(0) as u64,
        )
        .expect("a seam band narrower than a voxel's count");
        assert!(coverage.iter().all(|count| *count <= bound));
        let covered: u64 = coverage.iter().map(|count| u64::from(*count)).sum();
        assert_eq!(covered, rows.len() as u64);
    }

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
    #[test]
    fn the_video_arena_holds_all_tiles_of_one_clip() {
        let config = crate::video_vae::tests::video_config();
        let canvas = crate::video_vae::Canvas::clip(&config, 48, 84);
        let arena = video_arena(&config, &canvas).expect("the arena");
        // The served 768x1344 canvas at 256/64 is 4 rows of 7 tiles, each 16x16 latent voxels of
        // 7 frames.
        assert_eq!(canvas.frames, 7);
        assert_eq!(canvas.height, 48);
        assert_eq!(canvas.width, 84);
        let tiles = crate::video_vae::Tiles::of(&config, &canvas)
            .expect("the grid")
            .tiles(&config, &canvas)
            .len();
        assert_eq!(tiles, 28);
        let tokens = arena
            .iter()
            .find(|buffer| buffer.name == "stream")
            .expect("the residual stream");
        assert_eq!(tokens.elements, 28 * 1797 * 2048);
        assert_eq!(tokens.bytes(), tokens.elements * 2);
        // The rotary factors are the rotated width, not the head.
        let rotary = arena
            .iter()
            .find(|buffer| buffer.name == "rotary")
            .expect("the factors");
        assert_eq!(rotary.elements, 28 * 1797 * 48);
        // Every buffer is named once and no buffer is empty.
        let mut names: Vec<&str> = arena.iter().map(|buffer| buffer.name).collect();
        let count = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), count);
        assert!(arena.iter().all(|buffer| buffer.elements > 0));
    }

    #[test]
    fn a_tiles_placement_is_a_view_of_the_raster_and_its_blends_read_the_neighbour() {
        // The served canvas's output width: the raster the tiles are placed into.
        const WIDTH: u32 = 1344;
        // 768x1344 pixels: 48x84 latent positions, so the checkpoint's 256-pixel tiles and 64-pixel
        // minimum overlap give a grid of four columns and seven rows, each tile 16x16 latent.
        let config = crate::video_vae::tests::video_config();
        let canvas = Canvas::clip(&config, 48, 84);
        let clip = ClipDecoder::new(&config, &canvas, mean(), std()).expect("the output path");
        let tiles = Tiles::of(&config, &canvas)
            .expect("the grid")
            .tiles(&config, &canvas);
        assert_eq!(tiles.len(), 28);
        let frames = canvas.frames_out(&config);
        let dest = TensorView::new(
            ops::DevicePtr::new(0x1000),
            DType::F16,
            Shape::new(&[3, u64::from(frames), 768, u64::from(WIDTH)]),
        );
        let ratio = 16;
        let mut above_seams = 0;
        let mut left_seams = 0;
        for tile in &tiles {
            let first_row = u64::from(tile.latent_rows.0) * ratio;
            let first_column = u64::from(tile.latent_columns.0) * ratio;
            let view = clip
                .placement(dest, tile, first_row, first_column, 0, 0)
                .expect("the placement");
            // The block is the tile's canvas in pixels, and it sits where the seam table says.
            assert_eq!(view.shape.dims(), &[3, u64::from(frames), 256, 256]);
            let origin = (first_row * u64::from(WIDTH) + first_column) * 2;
            assert_eq!(view.ptr.address(), 0x1000 + origin);
            if tile.above != 0 {
                let previous = clip
                    .placement(dest, tile, first_row, first_column, tile.above, 0)
                    .expect("the row neighbour");
                assert_eq!(
                    previous.ptr.address(),
                    0x1000 + origin - u64::from(tile.above) * u64::from(WIDTH) * 2
                );
                assert!(clip.blend(tile.above, 2).is_ok());
                above_seams += 1;
            }
            if tile.left != 0 {
                let previous = clip
                    .placement(dest, tile, first_row, first_column, 0, tile.left)
                    .expect("the column neighbour");
                assert_eq!(
                    previous.ptr.address(),
                    0x1000 + origin - u64::from(tile.left) * 2
                );
                assert!(clip.blend(tile.left, 3).is_ok());
                left_seams += 1;
            }
        }
        // The canvas is 48 x 80 latent cells, so its grid is 4 rows of 7: every tile but the first row
        // blends upwards (3 x 7), and every tile but the first column blends left (4 x 6).
        assert_eq!(above_seams, 21);
        assert_eq!(left_seams, 24);
    }

    #[test]
    fn a_clip_needs_one_latent_view_and_one_position_table_per_tile() {
        let config = crate::video_vae::tests::video_config();
        let canvas = Canvas::clip(&config, 48, 84);
        let tiles = Tiles::of(&config, &canvas)
            .expect("the grid")
            .tiles(&config, &canvas);
        // 28 tiles of a seven-frame canvas: 7 x 256 voxel rows plus five register and tail rows.
        let rows: u64 = tiles
            .iter()
            .map(|tile| crate::video_vae::Canvas::rows(&tile.canvas))
            .sum();
        assert_eq!(rows, 28 * (7 * 256 + 5));
        let tokens = Shape::new(&[rows, 3072]);
        assert_eq!(
            check_clip(&tiles, 28, 28, tokens).expect("a whole clip"),
            rows
        );
        // A missing view or table is refused before any launch.
        assert!(check_clip(&tiles, 27, 28, tokens).is_err());
        assert!(check_clip(&tiles, 28, 27, tokens).is_err());
        // So is a token buffer that is not one row per token of the clip.
        assert!(check_clip(&tiles, 28, 28, Shape::new(&[rows - 1, 3072])).is_err());
        assert!(check_clip(&tiles, 28, 28, Shape::new(&[rows, 2048])).is_ok());
        assert!(check_clip(&tiles, 28, 28, Shape::new(&[rows])).is_err());
    }
}

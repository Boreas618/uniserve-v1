//! The H3 decoders' plan: how a video is spliced into clips, what a clip of decoded frames keeps, and
//! the layout each decoder hands the media stage.
//!
//! `uniserve/models/minimax_h3/decoding.py` and `output.py` are the reference. The video VAE decodes
//! one clip at a time and [`crate::output::VideoPostprocessor`] splices the clips into a timeline; the
//! audio VAE decodes one
//! channel-major stereo latent timeline. This module owns that arithmetic and the output layouts, and
//! nothing that touches a device: the packed latents, the decoded clip frames and the PCM are borrowed
//! views the worker hands in and out.
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
use crate::video_vae::{Canvas, Tile};
use crate::{
    Error, Result,
    config::{CLIP_FRAMES, CLIP_LATENTS},
};
use nn::layers::{TileGrid, TileStitch};
use nn::{Forward, Shape, TensorView};

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
/// The worker sizes the arena from these; the model never allocates. Names are the ones the arena
/// plan uses, so a mismatch is visible in one place - **and the order
/// is the order the model reserves them in**, because the arena is one cursor: a buffer listed before
/// another is also reserved before it, and anything the code reserves inside a loop is multiplied.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Buffer {
    /// The buffer's name, unique within its plan: the worker keys storage by it.
    pub name: String,
    /// Its element type: `f16`, `f32` or `u8`.
    pub element: &'static str,
    /// How many elements it holds.
    pub elements: u64,
}

impl Buffer {
    /// The width of one element, by the name the plans use.
    ///
    /// The names below are the whole set a plan may declare: the four the arena holds floats and bytes in,
    /// and `i32` for the index buffers a gather reads. `f16` and `bf16` are two bytes each, `i32` and `f32`
    /// are four, which is the difference between a 200 MiB buffer and a 400 MiB one. An unknown name is
    /// refused rather than assumed, because the assumption - four bytes - would be half the budget for a
    /// reduced-precision buffer.
    pub fn width(&self) -> Result<u64> {
        match self.element {
            "u8" => Ok(1),
            "f16" | "bf16" => Ok(2),
            "i32" | "f32" => Ok(4),
            other => Err(Error::config(format!(
                "{other} is not an element type a plan declares: u8, f16, bf16, i32 or f32"
            ))),
        }
    }

    /// The bytes the buffer occupies in an arena: its size rounded up to the arena's alignment.
    ///
    /// `plan::Arena::reserve` aligns every give-out to [`nn::forward::ALIGNMENT`], so a reservation
    /// summed from raw sizes is short by up to `ALIGNMENT - 1` bytes **per buffer** - multiplied by
    /// the number of buffers, which for the video decoder's sixteen is up to 4 KiB. The first run to
    /// meet it was short by 64 bytes, exactly the padding of the two rotary tables (`1797 x 24 x 4`
    /// is not a multiple of 256).
    pub fn reservation(&self) -> u64 {
        self.bytes().next_multiple_of(nn::forward::ALIGNMENT)
    }

    /// The bytes the buffer occupies.
    pub fn bytes(&self) -> u64 {
        // A plan with an unknown element name is caught by [`Self::width`]; this answers for the four
        // names the crate's plans use and treats anything else as four bytes so a *report* never panics.
        self.elements * self.width().unwrap_or(4)
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
/// `tiles` are in grid order (`Tiles::tiles`), and each tile contributes
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
/// The bytes a runner must reserve for `plan`: every buffer at the arena's alignment.
///
/// **A plan summed raw is short.** The arena aligns each give-out to [`nn::forward::ALIGNMENT`] and a
/// plan's entries are not all multiples of it, so the reservation a runner makes has to be the
/// aligned sum rather than the sum of the sizes. The failure this prevents is the smallest that has
/// ever stopped a stage: `22081536 bytes requested at offset 368378880 of a 390460352-byte region`,
/// 64 bytes past the end, because `1797 x 24 x 4` twice is 64 bytes of padding the sum left out.
pub fn arena_reservation(plan: &[Buffer]) -> u64 {
    plan.iter().map(Buffer::reservation).sum()
}

pub fn video_arena(
    config: &crate::VideoVaeConfig,
    canvas: &crate::video_vae::Canvas,
) -> Result<Vec<Buffer>> {
    use crate::video_vae::{ACTIVATION, STREAM};

    let tiles = crate::video_vae::Tiles::of(config, canvas)?;
    let grid = tiles.tiles(config, canvas);
    let ffn = u64::from(config.decoder_ffn_mult);
    let (rotated, _) = crate::video_vae::rotary(config)?;
    let columns = u64::from(config.decoder_num_attention_heads)
        * u64::from(config.decoder_attention_head_dim);
    let (height, width_px) = canvas.pixels(config);
    // **Every tile runs in the same launches**, so every buffer holds the grid's rows: summed, because
    // a tile at the canvas's edge can be narrower than the tile size.
    let rows: u64 = grid.iter().map(|tile| tile.canvas.rows()).sum();
    let voxels: u64 = grid.iter().map(|tile| tile.canvas.voxels()).sum();
    let tile = grid.first().ok_or_else(|| {
        Error::config(format!(
            "a canvas of {} x {} pixels has no tile to decode",
            height, width_px
        ))
    })?;
    let latent = u64::from(config.latent_channels);
    // **The order is the reservation order**, and the plan is live: a runner sizes the stream's graph
    // range from this sum, so a buffer the code reserves and this list does not state stops a capture.
    // There is no entry for the clip's raster: the stitch's destination is the caller's
    // `DecodedMedia.video`, which is the runner's buffer and not the arena's.
    let patch = u64::from(config.out_channels)
        * u64::from(config.temporal_compression())
        * u64::from(config.spatial_compression())
        * u64::from(config.spatial_compression());
    // **The element types are the module's table** (`video_vae.rs`): the stream and the restored
    // latents are wide, and everything a projection reads or writes is narrow.
    let narrow = crate::text_encoder::element_of(ACTIVATION);
    let wide = crate::text_encoder::element_of(STREAM);
    let buffer = |name: &str, element, elements| Buffer {
        name: name.to_owned(),
        element,
        elements,
    };
    Ok(vec![
        buffer("tokens", narrow, rows * patch),
        buffer("stream", wide, rows * columns),
        buffer("restored", wide, voxels * latent),
        buffer("rounded", narrow, voxels * latent),
        buffer("quantized", narrow, voxels * latent),
        buffer("projected", narrow, voxels * columns),
        // One tile's factors: every tile has the same positions, and the rotation repeats the table.
        buffer(
            "cosine",
            narrow,
            tile.canvas.rows() * u64::from(rotated / 2),
        ),
        buffer("sine", narrow, tile.canvas.rows() * u64::from(rotated / 2)),
        buffer("normalized", narrow, rows * columns),
        buffer("query_key_value", narrow, rows * 3 * columns),
        buffer("attended", narrow, rows * columns),
        buffer("update", narrow, rows * columns),
        buffer("gate_up", narrow, rows * 2 * columns * ffn),
        buffer("gated", narrow, rows * columns * ffn),
    ])
}

/// The video decoder's tiled output path: a clip's tokens become its raster in one launch.
///
/// The transformer leaves every tile's tokens in one buffer, and each pixel of the clip's raster is a
/// function of at most three of them (`ops::tile_stitch`), which is the reference's stitch
/// (`uniserve/nn/vae/tiled.py`, `stitch_tiles`) with the unpack in front of it: a tile's rows and
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

    /// The ImageNet constants as the model publishes them: host memory, one f32 per channel.
    #[test]
    fn a_plans_element_names_are_a_closed_set_with_stated_widths() {
        // A reduced-precision buffer counted as four bytes doubles its budget, and an unknown name used to
        // fall through to that same four, so the set is closed here and every width is stated. `i32` is the
        // index buffers a gather reads, four bytes like `f32` but not a float.
        let buffer = |element: &'static str| Buffer {
            name: format!("probe.{element}"),
            element,
            elements: 1024,
        };
        for (element, width) in [("u8", 1), ("f16", 2), ("bf16", 2), ("i32", 4), ("f32", 4)] {
            let buffer = buffer(element);
            assert_eq!(buffer.width().expect(element), width, "{element}");
            assert_eq!(buffer.bytes(), 1024 * width, "{element}");
        }
        // Anything else is refused rather than assumed, at the widths a budget depends on.
        assert!(buffer("f64").width().is_err());
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
    /// The plan is the block's own allocation list, so its element types are the decode's - all of
    /// them but the rotary tables, which the block allocates at F32 whatever it decodes at. A plan
    /// that spelled one type for both reserved half of what an FP32 decode asks the arena for, and the
    /// shortfall surfaced inside the clip as "cannot give out" rather than as a plan that was short.
    /// A reservation summed raw is short by every buffer's padding - and the buffers that pad are
    /// the ones whose element count is not a multiple of the alignment, which for this decoder is the
    /// two rotary tables. The first device run to meet it was 64 bytes short of a 372 MiB arena.
    #[test]
    fn a_plan_is_reserved_at_the_arenas_alignment_not_at_its_raw_sum() {
        let config = crate::video_vae::tests::video_config();
        let canvas = crate::video_vae::Canvas::clip(&config, 16, 16);
        let plan = video_arena(&config, &canvas).expect("the plan");
        let raw: u64 = plan.iter().map(Buffer::bytes).sum();
        let reserved = arena_reservation(&plan);
        assert_eq!(
            reserved,
            plan.iter()
                .map(|buffer| buffer.bytes().next_multiple_of(nn::forward::ALIGNMENT))
                .sum::<u64>(),
            "the reservation is each buffer at the arena's alignment"
        );
        assert!(reserved >= raw, "an aligned sum is never smaller");
        // **The difference is padding, buffer by buffer, and not a multiple of anything.** An earlier
        // version of this test asserted `(reserved - raw) % 256 == 0`, which is false whenever the
        // buffers' raw sum is not itself a multiple of the alignment - the paddings sum to whatever
        // they sum to. What is true is that each buffer pads by less than one alignment, and that the
        // total difference is those paddings.
        let padding: u64 = plan
            .iter()
            .map(|buffer| buffer.reservation() - buffer.bytes())
            .sum();
        assert_eq!(
            reserved - raw,
            padding,
            "the difference is the buffers' padding"
        );
        for buffer in &plan {
            assert!(
                buffer.reservation() - buffer.bytes() < nn::forward::ALIGNMENT,
                "{}: {} pads by no more than one alignment",
                buffer.name,
                buffer.bytes()
            );
        }
        // And the function's own rule, on a size no canvas has to supply: a buffer whose element
        // count is not a multiple of the alignment is rounded up, not down, and one that is a
        // multiple is left alone. The device run that found this was 64 bytes short because two
        // 172,512-byte tables were placed at their raw size.
        let uneven = Buffer {
            name: "a table".to_owned(),
            element: "f32",
            elements: 172_512 / 4,
        };
        assert_eq!(uneven.bytes(), 172_512);
        assert_eq!(uneven.reservation(), 172_544, "rounded up to the next 256");
        let even = Buffer {
            name: "an aligned tensor".to_owned(),
            element: "f32",
            elements: 65_536 / 4,
        };
        assert_eq!(even.reservation(), even.bytes(), "a multiple is not moved");
    }

    /// The plan is the block's own allocation list, so its element types are the module's table: the
    /// stream and the restored latents are FP32, and every other buffer is the activation's. A plan that
    /// spelled one type for both reserved half of what the stream asks the arena for, and the shortfall
    /// surfaced inside the capture as an overflow rather than as a plan that was short.
    #[test]
    fn the_video_arena_states_the_element_type_each_buffer_holds() {
        let config = crate::video_vae::tests::video_config();
        let canvas = crate::video_vae::Canvas::clip(&config, 16, 16);
        let plan = video_arena(&config, &canvas).expect("the plan");
        for buffer in &plan {
            let expected = match buffer.name.as_str() {
                "stream" | "restored" => "f32",
                _ => "f16",
            };
            assert_eq!(buffer.element, expected, "{}", buffer.name);
        }
    }

    #[test]
    fn the_video_arena_holds_every_tiles_rows() {
        let config = crate::video_vae::tests::video_config();
        let canvas = crate::video_vae::Canvas::clip(&config, 48, 84);
        let arena = video_arena(&config, &canvas).expect("the arena");
        // The served 768x1344 canvas at 256/64 is 4 rows of 7 tiles, each 16x16 latent voxels of
        // 7 frames.
        assert_eq!(canvas.frames, 7);
        let grid = crate::video_vae::Tiles::of(&config, &canvas)
            .expect("the grid")
            .tiles(&config, &canvas);
        assert_eq!(grid.len(), 28);
        // **Every tile runs in the same launches**, so the transformer's buffers hold the grid's rows;
        // only the rotary factors are one tile's, because the rotation repeats them.
        let tile_rows: u64 = grid[0].canvas.rows();
        let all_rows: u64 = grid.iter().map(|tile| tile.canvas.rows()).sum();
        let all_voxels: u64 = grid.iter().map(|tile| tile.canvas.voxels()).sum();
        let buffer = |name: &str| {
            arena
                .iter()
                .find(|buffer| buffer.name == name)
                .unwrap_or_else(|| panic!("the plan states {name}"))
        };
        let columns = 2048;
        // **The pixel count is the patch**: `out_channels × temporal_compression ×
        // spatial_compression²` is 3 · 4 · 16 · 16 = 3072 elements per latent voxel.
        let patch = u64::from(config.out_channels)
            * u64::from(config.temporal_compression())
            * u64::from(config.spatial_compression())
            * u64::from(config.spatial_compression());
        assert_eq!(buffer("tokens").elements, all_rows * patch);
        assert_eq!(buffer("stream").elements, all_rows * columns);
        assert_eq!(buffer("projected").elements, all_voxels * columns);
        assert_eq!(buffer("query_key_value").elements, all_rows * 3 * columns);
        assert_eq!(buffer("gate_up").elements, all_rows * 2 * columns * 4);
        assert_eq!(buffer("gated").elements, all_rows * columns * 4);
        let half = u64::from(crate::video_vae::rotary(&config).expect("the rotary").0 / 2);
        assert_eq!(buffer("cosine").elements, tile_rows * half);
        assert_eq!(buffer("sine").elements, tile_rows * half);
        // Every buffer is named once and no buffer is empty.
        let mut names: Vec<&str> = arena.iter().map(|buffer| buffer.name.as_str()).collect();
        let count = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), count);
        assert!(arena.iter().all(|buffer| buffer.elements > 0));
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

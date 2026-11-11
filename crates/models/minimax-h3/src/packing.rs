//! The host plan of one H3 request: its frames, clips and latents, the rows its packed sequence
//! holds, and the tile order a sparse layout walks them in.
//!
//! Nothing here touches a device: the
//! worker uploads the tables this module computes into the batches its kernels read, so a
//! captured graph addresses them unchanged from capture to replay.
//!
//! ## The timeline
//!
//! A generated video has `17n + 5` frames at 24 fps; the video VAE encodes each 17-frame clip to 5
//! latent frames and the final clip drops 3 while the next clip's overlap repeats them, so `n`
//! clips give `5n + 2` latent frames. The latents are packed as 2x2 patches, one row per patch per
//! latent frame, and the stereo audio is one row per channel per 40 Hz latent step.
//!
//! ## Two layouts
//!
//! A dense-attention network packs `[video | audio | text | conditions | padding]` in raster order.
//! A VSA-sparse network packs `[text | conditions | audio | video]` in whole 64- or 128-row tiles,
//! the generated video tile-major over its token grid, so each key tile the sparse attention keeps
//! is a whole tile of rows.

use std::ops::Range;

use proto::video::Canvas;

use processing::minimax_h3::{ConditionPacking, VisualRows, condition_tiles};

use crate::config::{
    AUDIO_CHANNELS, AUDIO_HOP, AttentionKind, CANVAS_RULE, FPS, FRAMES, SPARSE_TILE_ROWS,
};
use crate::{Error, Result};

/// The `(time, height, width)` token grid of one video tile of each tile size.
///
/// A tile is one video tile shape: dense condition rows fill whole tiles of the same row count, and
/// a sparse layout's tiles are these shapes.
pub const VIDEO_TILE_SHAPES: [(u32, (u32, u32, u32)); 2] = SPARSE_TILE_ROWS;

/// The token tags a packed row carries: which modality its modulation reads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i32)]
pub enum Tag {
    /// A generated or condition video row.
    Video = 0,
    /// A prompt row.
    Text = 1,
    /// A generated or reference audio row.
    Audio = 2,
}

/// The timestep group a packed row reads its modulation under.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i32)]
pub enum Group {
    /// The generated video's timestep.
    Video = 0,
    /// The generated audio's timestep.
    Audio = 1,
    /// The clean visual-condition timestep.
    VisualCondition = 2,
    /// The clean audio-reference timestep.
    AudioCondition = 3,
}

/// The video VAE's spatial compression and the DiT's patch side in latent cells, as request
/// processing states them for the canvas rule.
pub use processing::minimax_h3::{LATENT_COMPRESSION, PATCH_SIDE};

/// The `(height, width)` latent raster of a canvas.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LatentRaster {
    /// Latent rows: the canvas height over the VAE's 16x spatial compression.
    pub height: u32,
    /// Latent columns.
    pub width: u32,
}

impl LatentRaster {
    /// The latent raster of `canvas`: its sides over the VAE's spatial compression. A canvas off
    /// the canvas grid is refused by [`latent_raster`], which callers that do not hold a planned
    /// canvas go through.
    pub const fn of(canvas: &Canvas) -> Self {
        Self {
            height: canvas.height / LATENT_COMPRESSION,
            width: canvas.width / LATENT_COMPRESSION,
        }
    }

    /// Patch rows of one latent frame: one row per [`PATCH_SIDE`]-square patch.
    pub fn patch_rows(&self) -> u32 {
        (self.height / PATCH_SIDE) * (self.width / PATCH_SIDE)
    }
}

/// The `(time, height, width)` token grid of `frames` latent frames of a latent raster.
pub fn video_grid(frames: u32, raster: LatentRaster) -> (u32, u32, u32) {
    (
        frames,
        raster.height / PATCH_SIDE,
        raster.width / PATCH_SIDE,
    )
}

/// The latent raster of a canvas whose sides lie on the canvas rule's grid: whole patches of the
/// VAE's latent cells.
pub fn latent_raster(canvas: &Canvas) -> Result<LatentRaster> {
    let multiple = CANVAS_RULE.multiple;
    if !canvas.width.is_multiple_of(multiple) || !canvas.height.is_multiple_of(multiple) {
        return Err(Error::config(format!(
            "an H3 canvas side must be a multiple of {multiple} pixels, got {}x{}",
            canvas.width, canvas.height
        )));
    }
    Ok(LatentRaster::of(canvas))
}

/// The latent frames of a generated video of `num_frames` frames, refusing a count the frame rule
/// does not generate ([`FRAMES`]).
pub fn timeline_latents(num_frames: u32) -> Result<u32> {
    if !FRAMES.is_legal(num_frames) {
        return Err(Error::config(format!(
            "an H3 frame count is one of the frame rule {FRAMES:?}, got {num_frames}"
        )));
    }
    Ok(FRAMES.latent_frames(num_frames))
}

/// Audio latent steps of a generated frame count: the duration in 40 Hz latents, rounded to the
/// nearest step.
///
/// At 24 fps and 40 Hz the exact value is never halfway between two steps, so the rounding
/// direction is unambiguous; it is round-half-to-even because the reference uses Python's `round`.
///
/// **This is the request's direction, and it is not the encode's.** At 24 fps and 32 kHz a frame is
/// `4000 / 3` samples and a latent step is `800`, so 5 latents span 3 frames exactly and a request's
/// latent count is `5n / 3` rounded: its track is `800 * round(5n / 3)` samples against an exact
/// `4000n / 3`, and the two differ by a third of a hop (`800 / 3` = 266.67 samples) with the sign set
/// by `n mod 3`. A frame count of `0 mod 3` is exact; of `1 mod 3` the track is long (124 frames,
/// 207 latents, `165600` samples against `165333.33`); of `2 mod 3` it is **short** (107 frames, 178
/// latents, `142400` against `142666.67`). **So a request's track may be shorter than its video**, and
/// whoever muxes them states each stream's own length rather than deriving either from the duration.
///
/// [`processing::minimax_h3::audio_latents`] is the other direction - samples to latents, `ceil` - and
/// it is the one that can never be short, because its input is a sample count rather than a frame
/// count. The two keep distinct-looking names for that reason: one answers "what does this request
/// need", the other "what does this track encode to".
pub fn audio_latent_frames(num_frames: u32) -> u32 {
    let exact =
        f64::from(num_frames) * f64::from(crate::config::AUDIO_LATENTS_PER_SECOND) / f64::from(FPS);
    exact.round_ties_even() as u32
}

/// Audio rows of a generated frame count: one row per channel per latent step.
pub fn audio_rows(num_frames: u32) -> u32 {
    track_rows(audio_latent_frames(num_frames))
}

pub use processing::minimax_h3::track_rows;

/// Samples of one generated frame count's soundtrack.
pub fn audio_samples(num_frames: u32) -> u32 {
    audio_latent_frames(num_frames) * AUDIO_HOP
}

/// What one condition contributes to the packed sequence.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SegmentKind {
    /// One latent frame of the target canvas anchoring the first generated frame.
    FirstFrame,
    /// One latent frame of the target canvas anchoring the last generated frame.
    LastFrame,
    /// One latent frame of a reference image at its own raster.
    Image,
    /// A reference video's latent frames, with its soundtrack when it has one.
    Video,
    /// A reference track's stereo audio rows.
    Audio,
}

impl SegmentKind {
    /// Whether the condition anchors a generated frame.
    pub const fn is_keyframe(self) -> bool {
        matches!(self, Self::FirstFrame | Self::LastFrame)
    }
}

/// One condition's rows in the packed sequence.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Segment {
    /// What the condition contributes.
    pub kind: SegmentKind,
    /// The condition's position in request order.
    pub condition: u32,
    /// Latent frames of its pixels: one for a still, none for an audio-only reference.
    pub frames: u32,
    /// The latent raster of one of its frames; empty for an audio-only reference.
    pub raster: LatentRaster,
    /// Audio latent steps the condition carries.
    pub audio_frames: u32,
}

impl Segment {
    /// Visual rows: one per 2x2 patch of every latent frame.
    pub fn video_rows(&self) -> u32 {
        self.frames * self.raster.patch_rows()
    }

    /// Audio rows: channel-major stereo ([`track_rows`]).
    pub fn audio_rows(&self) -> u32 {
        track_rows(self.audio_frames)
    }

    /// The condition's visual rows as the tiling counts them ([`condition_tiles`]).
    fn visual(&self) -> VisualRows {
        match self.kind {
            SegmentKind::FirstFrame | SegmentKind::LastFrame | SegmentKind::Image => {
                VisualRows::Patches(self.video_rows())
            }
            SegmentKind::Video => {
                let (frames, rows, columns) = video_grid(self.frames, self.raster);
                VisualRows::Grid([frames, rows, columns])
            }
            SegmentKind::Audio => VisualRows::None,
        }
    }
}

/// The tile shape of a sparse layout of `tile` rows.
pub fn tile_shape(tile: u32) -> Result<(u32, u32, u32)> {
    VIDEO_TILE_SHAPES
        .iter()
        .find_map(|(rows, shape)| (*rows == tile).then_some(*shape))
        .ok_or_else(|| {
            Error::config(format!(
                "an H3 sparse tile holds {} rows, got {tile}",
                VIDEO_TILE_SHAPES
                    .iter()
                    .map(|(rows, _)| rows.to_string())
                    .collect::<Vec<_>>()
                    .join(" or ")
            ))
        })
}

/// How a layout of `attention` tiles its condition region, as request processing counts it: whole
/// tiles of the attention tile under sparse attention, row for row under dense attention.
/// `capacity` is the deployment's condition capacity.
///
/// # Errors
///
/// A sparse tile size the video grid does not hold.
pub fn condition_packing(attention: AttentionKind, capacity: u32) -> Result<ConditionPacking> {
    let (tile_rows, (time, height, width)) = match attention {
        AttentionKind::Sparse { tile, .. } => (tile, tile_shape(tile)?),
        AttentionKind::Dense => (1, (1, 1, 1)),
    };
    Ok(ConditionPacking {
        capacity,
        tile_rows,
        video_tile: [time, height, width],
    })
}

/// Rows `segments` take in a condition region of `packing`: their tiles' rows
/// ([`condition_tiles`], the one statement request processing counts by).
pub fn condition_rows(segments: &[Segment], packing: &ConditionPacking) -> u32 {
    segments
        .iter()
        .map(|segment| {
            condition_tiles(segment.audio_rows(), segment.visual(), packing)
                .saturating_mul(packing.tile_rows.max(1))
        })
        .fold(0, u32::saturating_add)
}

/// Tiles of a segment of `tiles` tiles that a query attends, keeping `1 - sparsity` of them.
///
/// `ceil((1 - sparsity) * tiles)`, at least one and at most every tile, in the reference's
/// double-precision arithmetic.
pub fn kept_tiles(sparsity: f64, tiles: u32) -> u32 {
    let keep = (1.0 - sparsity) * f64::from(tiles);
    (keep.ceil() as u32).clamp(1, tiles.max(1))
}

/// The order a `(time, height, width)` token grid is walked in whole tiles of `shape`:
/// `(raster, sizes)`, the raster row (row-major over the grid) of every row in tile-major order,
/// and the valid rows of each tile.
///
/// Tiles cover the grid time-major, then by height, then by width; the grid's far edges cut
/// boundary tiles short. Within a tile the rows inside the grid follow in row-major order, packed at
/// the tile's front.
pub fn tile_order(grid: (u32, u32, u32), shape: (u32, u32, u32)) -> (Vec<u32>, Vec<u32>) {
    let (frames, height, width) = grid;
    let (tile_t, tile_h, tile_w) = shape;
    let across = (
        frames.div_ceil(tile_t),
        height.div_ceil(tile_h),
        width.div_ceil(tile_w),
    );
    let tiles = across.0 * across.1 * across.2;
    let rows_in_tile = tile_t * tile_h * tile_w;
    let mut raster = Vec::with_capacity((rows_in_tile * tiles) as usize);
    let mut sizes = Vec::with_capacity(tiles as usize);
    for tile in 0..tiles {
        let tile_time = tile / (across.1 * across.2);
        let tile_height = tile / across.2 % across.1;
        let tile_width = tile % across.2;
        let start = raster.len();
        for offset in 0..rows_in_tile {
            let time = tile_time * tile_t + offset / (tile_h * tile_w);
            let row = tile_height * tile_h + offset / tile_w % tile_h;
            let column = tile_width * tile_w + offset % tile_w;
            if time < frames && row < height && column < width {
                raster.push((time * height + row) * width + column);
            }
        }
        sizes.push((raster.len() - start) as u32);
    }
    (raster, sizes)
}

/// Rows of the first `tiles` tiles of [`tile_order`], without ordering the rows.
///
/// `tiles` is clamped to the grid's tile count. The whole tile slabs of frames count first, then
/// the whole tile rows of the next slab, then the leading tiles of its next tile row.
pub fn tiled_rows(grid: (u32, u32, u32), shape: (u32, u32, u32), tiles: u32) -> u32 {
    let (frames, height, width) = grid;
    let (tile_t, tile_h, tile_w) = shape;
    let across_h = height.div_ceil(tile_h);
    let across_w = width.div_ceil(tile_w);
    let total = frames.div_ceil(tile_t) * across_h * across_w;
    let taken = tiles.min(total);
    let (slabs, rest) = (taken / (across_h * across_w), taken % (across_h * across_w));
    let (lines, columns) = (rest / across_w, rest % across_w);
    let mut rows = (slabs * tile_t).min(frames) * height * width;
    if rest > 0 {
        let depth = tile_t.min(frames - slabs * tile_t);
        rows += depth * (lines * tile_h).min(height) * width;
        rows += depth * tile_h.min(height - lines * tile_h) * (columns * tile_w).min(width);
    }
    rows
}

/// Rotary time of one latent frame's gap: the output frames it covers, rescaled.
const ROPE_FRAME_RESCALE: f64 = 5.0 / 3.0;

/// Output frames each of the five latent frames of one clip covers.
const ROPE_FRAMES_PER_LATENT: [u32; 5] = [1, 4, 4, 4, 4];

/// Scale of the spatial rotary axes.
const ROPE_SPATIAL_SCALE: f64 = 32.0;

/// The rotary coordinates of the patches along one spatial axis.
///
/// The `latent / 2` coordinates are evenly spaced over an interval of length `extent /
/// sqrt_area`, centred on 0.5, so both axes share the scale of the raster's geometric-mean side,
/// and are then multiplied by [`ROPE_SPATIAL_SCALE`].
fn spatial_grid(extent: u32, sqrt_area: f64) -> Vec<f64> {
    let ratio = f64::from(extent) / sqrt_area;
    let left = (1.0 - ratio) / 2.0;
    let count = extent / 2;
    let step = ratio / f64::from(count);
    (0..count)
        .map(|index| (left + f64::from(index) * step) * ROPE_SPATIAL_SCALE)
        .collect()
}

/// One latent frame's patch coordinates, `(height, width)` per patch in raster order, and the
/// width axis a caller places audio at.
fn frame_grid(raster: LatentRaster) -> (Vec<[f64; 2]>, Vec<f64>) {
    let sqrt_area = f64::from(raster.height * raster.width).sqrt();
    let heights = spatial_grid(raster.height, sqrt_area);
    let widths = spatial_grid(raster.width, sqrt_area);
    let mut spatial = Vec::with_capacity(heights.len() * widths.len());
    for height in &heights {
        for width in &widths {
            spatial.push([*height, *width]);
        }
    }
    (spatial, widths)
}

/// The rotary times of `count` latent frames from `origin`.
///
/// The gap after a latent frame is [`ROPE_FRAME_RESCALE`] times the output frames it covers,
/// cycling through [`ROPE_FRAMES_PER_LATENT`].
fn temporal_grid(count: u32, origin: f64) -> Vec<f64> {
    let mut values = Vec::with_capacity(count as usize);
    let mut time = origin;
    for index in 0..count {
        values.push(time);
        time += ROPE_FRAME_RESCALE * f64::from(ROPE_FRAMES_PER_LATENT[index as usize % 5]);
    }
    values
}

/// Rotary time `frames` latent frames occupy.
///
/// The reference sums this series sequentially for a reference video's clock and pairwise for a
/// last keyframe's anchor; the two differ in the last place of FP64 from 16 latent frames on. This
/// module sums sequentially.
fn video_span(frames: u32) -> f64 {
    (0..frames)
        .map(|index| ROPE_FRAME_RESCALE * f64::from(ROPE_FRAMES_PER_LATENT[index as usize % 5]))
        .sum()
}

/// Place `frames` latent frames of `spatial` from `origin`.
fn place_frames(
    positions: &mut [[f64; 3]],
    rows: Range<u32>,
    frames: u32,
    origin: f64,
    spatial: &[[f64; 2]],
) {
    let times = temporal_grid(frames, origin);
    let mut index = rows.start as usize;
    for time in &times {
        for point in spatial {
            positions[index] = [*time, point[0], point[1]];
            index += 1;
        }
    }
    debug_assert_eq!(index, rows.end as usize);
}

/// Place one channel-major stereo block from `origin`.
///
/// Audio advances one unit per latent; it has no height coordinate, and the two channels sit at
/// the two ends of `width_grid`.
fn place_audio(
    positions: &mut [[f64; 3]],
    rows: Range<u32>,
    frames: u32,
    origin: f64,
    width_grid: &[f64],
) {
    let mut index = rows.start as usize;
    for channel in 0..AUDIO_CHANNELS {
        let width = if channel == 0 {
            width_grid[0]
        } else {
            width_grid[width_grid.len() - 1]
        };
        for frame in 0..frames {
            positions[index] = [origin + f64::from(frame), 0.0, width];
            index += 1;
        }
    }
    debug_assert_eq!(index, rows.end as usize);
}

/// Place a keyframe on the generated timeline of `frames` latent frames.
///
/// A keyframe anchoring the first generated frame shares its first latent frame's time and one
/// anchoring the last frame its last one's, both at the target's spatial grid.
fn place_keyframe(
    positions: &mut [[f64; 3]],
    rows: Range<u32>,
    keyframe: &Segment,
    origin: f64,
    frames: u32,
    spatial: &[[f64; 2]],
) {
    let time = match keyframe.kind {
        SegmentKind::FirstFrame => origin,
        _ => origin + video_span(frames) - ROPE_FRAME_RESCALE,
    };
    for (offset, point) in spatial.iter().enumerate() {
        positions[rows.start as usize + offset] = [time, point[0], point[1]];
    }
    debug_assert_eq!(rows.start as usize + spatial.len(), rows.end as usize);
}

/// Row extents of one dense layout: `[video | audio | prefix | padding]`.
///
/// The generated video rows (raster order: latent frame, then patch row, then patch column) start
/// at row 0 and the generated stereo audio rows (channel-major) follow them, so a sequence shard's
/// generated rows depend on the layout alone. The prefix segment holds a request's text rows and
/// then its condition rows, contiguous; the rest of the segment and the alignment rows are
/// padding, which attention excludes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DensePacking {
    /// Generated video rows.
    pub video_rows: u32,
    /// Generated audio rows, both stereo channels.
    pub audio_rows: u32,
    /// Text capacity of the prefix segment.
    pub text_rows: u32,
    /// Condition capacity of the prefix segment.
    pub condition_rows: u32,
    /// All rows, a multiple of the layout's alignment.
    pub padded_tokens: u32,
}

impl DensePacking {
    /// First row of the prefix segment.
    pub fn prefix_start(&self) -> u32 {
        self.video_rows + self.audio_rows
    }

    /// Row of the prefix source that holds zeros ([`nn::capability::PrefixSource::zero_row`]).
    pub const fn zero_row(&self) -> u32 {
        nn::capability::PrefixSource {
            text_rows: self.text_rows,
            condition_rows: self.condition_rows,
        }
        .zero_row()
    }
}

/// Size the dense layout of one frame count, canvas and prefix capacity.
///
/// `token_multiple` aligns the padded row count so that every sequence rank holds an equal shard.
pub fn dense_packing(
    num_frames: u32,
    canvas: &Canvas,
    text_rows: u32,
    condition_rows: u32,
    token_multiple: u32,
) -> Result<DensePacking> {
    if text_rows < 1 || token_multiple < 1 {
        return Err(Error::config(
            "a dense layout needs positive text capacity and a positive alignment".to_owned(),
        ));
    }
    let raster = latent_raster(canvas)?;
    let frames = timeline_latents(num_frames)?;
    let video_rows = frames * raster.patch_rows();
    let audio_rows = audio_rows(num_frames);
    let rows = video_rows + audio_rows + text_rows + condition_rows;
    Ok(DensePacking {
        video_rows,
        audio_rows,
        text_rows,
        condition_rows,
        padded_tokens: rows.div_ceil(token_multiple) * token_multiple,
    })
}

/// One request's row tables in its layout, dense or sparse.
///
/// The coordinates are the released layout: text row `i` sits at `(i, 0, 0)` and the media clock
/// starts at the prompt length. References advance it in packed order: an image by one unit at its
/// own spatial grid; an audio track by its latent count, its channels at the ends of the target
/// width grid; a video by the longer of its soundtrack and its frames. The generated video and
/// audio start where the references leave the clock.
///
/// A sparse layout holds the same rows in tiles: a keyframe's rows fill whole dense tiles, as an
/// image's do after a reference's audio rows, a reference video forms its own segment of video
/// tiles over its token grid, and the generated video forms the last segment. Its tile tables are
/// empty for a dense layout.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct RowTables {
    /// `[padded_tokens, 3]` rotary coordinates; rows outside the request hold zeros.
    pub position_ids: Vec<[f64; 3]>,
    /// `[padded_tokens]` modulation tags of [`Tag`]; padding rows carry [`Tag::Video`].
    pub token_tags: Vec<i32>,
    /// `[padded_tokens]` timestep groups of [`Group`].
    pub groups: Vec<i32>,
    /// `[padded_tokens]` row of the prefix source each packed row takes: text rows first, then
    /// condition rows, and the source's zero row for generated and padding rows.
    pub prefix_index: Vec<i32>,
    /// Rows attention attends in a dense layout: generated rows, text and conditions.
    pub used: u32,
    /// `[video rows]` packed row of each generated video row of a sparse layout, ascending.
    pub video_indices: Vec<u32>,
    /// `[video rows]` raster row of each entry of `video_indices`.
    pub video_raster_indices: Vec<u32>,
    /// `[tiles]` rows of each tile; zero for a tile the request leaves empty.
    pub variable_block_sizes: Vec<u32>,
    /// `[tiles]` video segment of each tile, `-1` for a dense or empty tile.
    pub tile_segments: Vec<i32>,
    /// `[tiles]` tiles below each segment index: the dense and empty tiles included.
    pub segment_starts: Vec<i32>,
    /// `[tiles]` key tiles a video query keeps of each segment.
    pub segment_keep: Vec<i32>,
}

/// Build one request's tables in its dense layout.
pub fn dense_tables(
    packing: &DensePacking,
    num_frames: u32,
    canvas: &Canvas,
    num_text_tokens: u32,
    segments: &[Segment],
    vision_spans: &[(u32, u32)],
) -> Result<RowTables> {
    let condition_rows = condition_rows(segments, &condition_packing(AttentionKind::Dense, 0)?);
    // One refusal per subject, and each names its own count: the two limits are independent - a
    // prompt's rows and a case's condition rows - and a single sentence covering both makes a caller
    // compare two numbers against a message that does not say which of them was too large.
    if num_text_tokens < 1 || num_text_tokens > packing.text_rows {
        return Err(Error::config(format!(
            "a prompt of {num_text_tokens} row(s) does not fit the {} text rows of the layout's prefix \
             segment",
            packing.text_rows
        )));
    }
    if condition_rows > packing.condition_rows {
        return Err(Error::config(format!(
            "the case's {condition_rows} condition row(s) do not fit the {} the layout's prefix \
             segment reserves: the reservation is the task's own, and a `t2va` deployment reserves none",
            packing.condition_rows
        )));
    }
    if vision_spans
        .iter()
        .any(|(start, stop)| !(start < stop && *stop <= num_text_tokens))
    {
        return Err(Error::config(
            "vision spans must lie within the prompt".to_owned(),
        ));
    }
    let raster = latent_raster(canvas)?;
    let rows = packing.padded_tokens;
    let frames = timeline_latents(num_frames)?;
    let audio_frames = audio_latent_frames(num_frames);
    let (spatial, width_grid) = frame_grid(raster);

    let mut positions = vec![[0.0_f64; 3]; rows as usize];
    let mut tags = vec![Tag::Video as i32; rows as usize];
    let mut groups = vec![Group::Video as i32; rows as usize];
    let mut prefix_index = vec![packing.zero_row() as i32; rows as usize];

    // The prompt: text row i at (i, 0, 0), its vision tokens tagged video.
    let text = packing.prefix_start()..packing.prefix_start() + num_text_tokens;
    for index in 0..num_text_tokens {
        positions[(text.start + index) as usize] = [f64::from(index), 0.0, 0.0];
        prefix_index[(text.start + index) as usize] = index as i32;
        tags[(text.start + index) as usize] = Tag::Text as i32;
    }
    for (start, stop) in vision_spans {
        for row in text.start + start..text.start + stop {
            tags[row as usize] = Tag::Video as i32;
        }
    }

    // Condition rows follow the text in packed order and gather the prefix source rows after its
    // text capacity. The media clock starts at the prompt length; references advance it in packed
    // order, and the generated video and audio start where they leave it. Keyframes are placed
    // once that origin is known.
    let mut cursor = text.end;
    let mut clock = f64::from(num_text_tokens);
    let mut keyframes: Vec<(Segment, Range<u32>)> = Vec::new();
    for (offset, segment) in segments.iter().enumerate() {
        let audio = cursor..cursor + segment.audio_rows();
        let visual = audio.end..audio.end + segment.video_rows();
        cursor = visual.end;
        for row in audio.clone() {
            tags[row as usize] = Tag::Audio as i32;
            groups[row as usize] = Group::AudioCondition as i32;
        }
        for row in visual.clone() {
            tags[row as usize] = Tag::Video as i32;
            groups[row as usize] = Group::VisualCondition as i32;
        }
        // The prefix source row of each condition row: the text capacity's first row plus the
        // rows of the conditions before it.
        let base = packing.text_rows
            + segments[..offset]
                .iter()
                .map(|earlier| earlier.video_rows() + earlier.audio_rows())
                .sum::<u32>();
        for (index, row) in (audio.start..visual.end).enumerate() {
            prefix_index[row as usize] = (base + index as u32) as i32;
        }
        if segment.kind.is_keyframe() {
            // Placed once the generated timeline's origin is known.
            keyframes.push((*segment, visual));
        } else if segment.kind == SegmentKind::Image {
            let (grid, _) = frame_grid(segment.raster);
            positions[visual.start as usize][0] = clock;
            for (offset, point) in grid.iter().enumerate() {
                positions[visual.start as usize + offset] = [clock, point[0], point[1]];
            }
            // An image takes one integer slot, not a latent frame's span.
            clock += 1.0;
        } else if segment.kind == SegmentKind::Audio {
            place_audio(
                &mut positions,
                audio,
                segment.audio_frames,
                clock,
                &width_grid,
            );
            clock += f64::from(segment.audio_frames);
        } else {
            let (grid, own_width) = frame_grid(segment.raster);
            place_audio(
                &mut positions,
                audio,
                segment.audio_frames,
                clock,
                &own_width,
            );
            place_frames(&mut positions, visual, segment.frames, clock, &grid);
            clock += f64::from(segment.audio_frames).max(video_span(segment.frames));
        }
    }
    for (segment, visual) in keyframes {
        place_keyframe(&mut positions, visual, &segment, clock, frames, &spatial);
    }

    // The generated video and audio start where the references left the clock.
    place_frames(
        &mut positions,
        0..packing.video_rows,
        frames,
        clock,
        &spatial,
    );
    let audio = packing.video_rows..packing.prefix_start();
    place_audio(
        &mut positions,
        audio.clone(),
        audio_frames,
        clock,
        &width_grid,
    );
    for row in audio {
        tags[row as usize] = Tag::Audio as i32;
        groups[row as usize] = Group::Audio as i32;
    }
    Ok(RowTables {
        position_ids: positions,
        token_tags: tags,
        groups,
        prefix_index,
        used: cursor,
        ..RowTables::default()
    })
}

/// Order a request's conditions as the packed sequence holds them.
///
/// Keyframes come first, in request order, then the references in request
/// order: the released `fl2va` layout holds only keyframes, and a `ref2va`
/// request's keyframes lead its references as the SGLang layout holds them.
/// A one-frame reference is an image; a longer one is a video, with its
/// soundtrack when it carries audio; a reference without pixels is audio.
///
/// A condition is stated as the denoiser receives it: its role, the raster and frames its pixels
/// are encoded at (none for an audio reference, whose raster is empty) and the samples of its
/// soundtrack.
pub fn condition_segments(
    conditions: &[nn::ConditionShape],
    canvas: &Canvas,
) -> Result<Vec<Segment>> {
    let mut keyframes = Vec::new();
    let mut references = Vec::new();
    for (index, condition) in conditions.iter().enumerate() {
        let visual = if condition.width == 0 || condition.height == 0 {
            None
        } else {
            {
                let raster = latent_raster(&Canvas {
                    width: condition.width,
                    height: condition.height,
                })?;
                // A condition is cut into clips, so its latent count is the frame rule's for any
                // count ([`processing::FrameClips::latent_frames`]): a still is one frame and one
                // latent frame, and a 30-frame reference is 7 latent frames, where rounding it up
                // to the next generated count (39) would give 12.
                if condition.frames == 0 {
                    return Err(Error::config(
                        "a video condition of no frame encodes nothing".to_owned(),
                    ));
                }
                Some((FRAMES.latent_frames(condition.frames), raster))
            }
        };
        let audio_frames =
            processing::minimax_h3::audio_latents(u64::from(condition.audio_samples)) as u32;
        let kind = match (condition.role, &visual) {
            (nn::ConditionRole::Reference, None) => SegmentKind::Audio,
            (nn::ConditionRole::Reference, Some((1, _))) => SegmentKind::Image,
            (nn::ConditionRole::Reference, Some(_)) => SegmentKind::Video,
            (role, Some((_, raster))) => {
                if *raster != latent_raster(canvas)? {
                    return Err(Error::config(
                        "H3 keyframes are fitted to the target canvas".to_owned(),
                    ));
                }
                if role == nn::ConditionRole::FirstFrame {
                    SegmentKind::FirstFrame
                } else {
                    SegmentKind::LastFrame
                }
            }
            (_, None) => {
                return Err(Error::config("an H3 keyframe carries its frame".to_owned()));
            }
        };
        if kind == SegmentKind::Image && audio_frames > 0 {
            return Err(Error::config(
                "an H3 image reference carries no audio".to_owned(),
            ));
        }
        let (frames, raster) = visual.unwrap_or((
            0,
            LatentRaster {
                height: 0,
                width: 0,
            },
        ));
        let segment = Segment {
            kind,
            condition: index as u32,
            frames,
            raster,
            audio_frames,
        };
        if kind.is_keyframe() {
            keyframes.push(segment);
        } else {
            references.push(segment);
        }
    }
    keyframes.extend(references);
    Ok(keyframes)
}

/// Row and tile counts of one sparse segment layout.
///
/// `[text | conditions | target audio | target video | padding]`, each whole tiles of `tile` rows.
/// Text and conditions are capacities that a request fills from their start; the generated audio
/// and video rows sit at positions the layout fixes. The generated video's rows are tile-major
/// over its token grid and its audio rows channel-major.
///
/// The counts follow from the frame count, the canvas, the capacities and the alignment alone, so
/// evaluating a layout builds no tables; [`segment_tables`] adds the tables of the same rows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SegmentExtent {
    /// Rows of one tile, a key of [`VIDEO_TILE_SHAPES`].
    pub tile: u32,
    /// Text capacity, whole tiles.
    pub text_rows: u32,
    /// Condition capacity, whole tiles.
    pub condition_rows: u32,
    /// Generated audio rows, both stereo channels.
    pub audio_rows: u32,
    /// `(time, height, width)` token grid of the generated video.
    pub video_grid: (u32, u32, u32),
    /// Generated video tiles, partial tiles included.
    pub video_tiles: u32,
    /// All rows, a multiple of the layout's alignment.
    pub padded_tokens: u32,
}

impl SegmentExtent {
    /// First row of the generated audio tiles.
    pub fn audio_start(&self) -> u32 {
        self.text_rows + self.condition_rows
    }

    /// First row of the generated video tiles.
    pub fn video_start(&self) -> u32 {
        self.audio_start() + self.audio_rows.div_ceil(self.tile) * self.tile
    }

    /// The prefix source's row of zeros ([`nn::capability::PrefixSource::zero_row`]).
    pub const fn zero_row(&self) -> u32 {
        nn::capability::PrefixSource {
            text_rows: self.text_rows,
            condition_rows: self.condition_rows,
        }
        .zero_row()
    }

    /// Generated audio rows packed before packed row `row`.
    pub fn audio_rows_below(&self, row: u32) -> u32 {
        row.saturating_sub(self.audio_start()).min(self.audio_rows)
    }

    /// Generated video rows packed before `row`, which must be a tile boundary of the video tiles.
    pub fn video_rows_below(&self, row: u32) -> Result<u32> {
        let shape = tile_shape(self.tile)?;
        let offset = i64::from(row) - i64::from(self.video_start());
        if offset % i64::from(self.tile) != 0 {
            return Err(Error::config(format!(
                "generated video rows are counted below tile boundaries, not row {row}"
            )));
        }
        let tiles = (offset / i64::from(self.tile)).clamp(0, i64::from(self.video_tiles));
        Ok(tiled_rows(self.video_grid, shape, tiles as u32))
    }

    /// Leading tiles that every request keeps dense or empty: the text tiles, and without
    /// conditions the audio tiles after them.
    pub fn dense_prefix_tiles(&self) -> u32 {
        let stop = if self.condition_rows == 0 {
            self.video_start()
        } else {
            self.text_rows
        };
        stop / self.tile
    }

    /// The key tiles each tile's queries attend, given the generated video's sparsity.
    ///
    /// Dense tiles attend every tile, and so may any tile before the generated video, which can be
    /// a reference segment. A generated video query attends the live dense tiles, which all precede
    /// it, and [`kept_tiles`] of its own tiles. Tiles after the video are empty.
    pub fn key_bounds(&self, sparsity: f64) -> Vec<u32> {
        let tiles = self.padded_tokens / self.tile;
        let start = self.video_start() / self.tile;
        let video = self.video_tiles;
        let keep = tiles.min(start + kept_tiles(sparsity, video));
        let mut bounds = vec![tiles; start as usize];
        bounds.extend(std::iter::repeat_n(keep, video as usize));
        bounds.extend(std::iter::repeat_n(1, (tiles - start - video) as usize));
        bounds
    }
}

/// Size the segment layout of one frame count, canvas and capacity.
///
/// `token_multiple`, a multiple of `tile`, aligns the padded row count so that every sequence rank
/// holds an equal shard of whole tiles.
pub fn segment_extent(
    num_frames: u32,
    canvas: &Canvas,
    text_rows: u32,
    condition_rows: u32,
    tile: u32,
    token_multiple: u32,
) -> Result<SegmentExtent> {
    // A segment layout is built from whole tiles: the tile size is one the video grid holds, and the
    // text, condition and alignment capacities are whole tiles of it. The refusal names the value
    // that failed rather than the five conditions, because they point at three different capacities
    // of a deployment - and a reader sent to all three has to find the four numbers to tell them
    // apart, which is what a partial run cannot show.
    let failing = if let Err(error) = tile_shape(tile) {
        error.to_string()
    } else if text_rows < tile || !text_rows.is_multiple_of(tile) {
        format!("the {text_rows} text rows are not whole {tile}-row tiles")
    } else if !condition_rows.is_multiple_of(tile) {
        format!("the {condition_rows} condition rows are not whole {tile}-row tiles")
    } else if token_multiple < tile || !token_multiple.is_multiple_of(tile) {
        format!("the {token_multiple}-row alignment is not a whole number of {tile}-row tiles")
    } else {
        String::new()
    };
    if !failing.is_empty() {
        return Err(Error::config(format!(
            "a sparse layout of this case cannot be built: {failing}"
        )));
    }
    let raster = latent_raster(canvas)?;
    let grid = video_grid(timeline_latents(num_frames)?, raster);
    let shape = tile_shape(tile)?;
    let video_tiles =
        grid.0.div_ceil(shape.0) * grid.1.div_ceil(shape.1) * grid.2.div_ceil(shape.2);
    let audio_rows = audio_rows(num_frames);
    let video_start = text_rows + condition_rows + audio_rows.div_ceil(tile) * tile;
    let end = video_start + video_tiles * tile;
    Ok(SegmentExtent {
        tile,
        text_rows,
        condition_rows,
        audio_rows,
        video_grid: grid,
        video_tiles,
        padded_tokens: end.div_ceil(token_multiple) * token_multiple,
    })
}

impl SegmentExtent {
    /// `[audio rows]` packed rows of the generated audio, ascending.
    pub fn audio_indices(&self) -> Range<u32> {
        let start = self.audio_start();
        start..start + self.audio_rows
    }
}

/// The rows of a dense segment of `count` rows filling tiles from `first_tile`.
///
/// The tiles are whole except the last, which holds the remainder; `count` must be positive.
fn dense_rows(first_tile: u32, count: u32, tile: u32, block_sizes: &mut [u32]) -> Range<u32> {
    let used = count.div_ceil(tile);
    for index in 0..used {
        block_sizes[(first_tile + index) as usize] = tile;
    }
    block_sizes[(first_tile + used - 1) as usize] = count - (used - 1) * tile;
    first_tile * tile..first_tile * tile + count
}

/// One video segment's tiles from `first_tile`.
///
/// Answers the packed row and the raster row of each entry, in tile-major order.
fn video_segment(
    first_tile: u32,
    grid: (u32, u32, u32),
    shape: (u32, u32, u32),
    tile: u32,
    block_sizes: &mut [u32],
    tile_segments: &mut [i32],
    tile_counts: &mut Vec<u32>,
) -> (Vec<u32>, Vec<u32>) {
    let (raster, sizes) = tile_order(grid, shape);
    let mut packed = Vec::with_capacity(raster.len());
    for (index, size) in sizes.iter().enumerate() {
        let slot = first_tile as usize + index;
        block_sizes[slot] = *size;
        tile_segments[slot] = tile_counts.len() as i32;
        let start = first_tile * tile + index as u32 * tile;
        packed.extend(start..start + size);
    }
    tile_counts.push(sizes.len() as u32);
    (packed, raster)
}

/// Place tile-major video rows from the raster row each packed row holds.
fn place_grid_frames(
    positions: &mut [[f64; 3]],
    rows: &[u32],
    raster: &[u32],
    frames: u32,
    origin: f64,
    spatial: &[[f64; 2]],
) {
    let times = temporal_grid(frames, origin);
    let per_frame = spatial.len() as u32;
    for (row, index) in rows.iter().zip(raster) {
        let point = &spatial[(*index % per_frame) as usize];
        positions[*row as usize] = [times[(*index / per_frame) as usize], point[0], point[1]];
    }
}

/// Build one request's tables in its segment layout `extent`.
///
/// `sparsity` is the fraction of a generated video segment's tiles the selection drops, and
/// `reference_keep` the fraction of a reference segment's tiles it keeps; the latter is required
/// the moment the request has a reference video, and unused otherwise.
pub fn segment_tables(
    extent: &SegmentExtent,
    num_frames: u32,
    canvas: &Canvas,
    (num_text_tokens, vision_spans): (u32, &[(u32, u32)]),
    segments: &[Segment],
    (sparsity, reference_keep): (f64, Option<f64>),
) -> Result<RowTables> {
    let tile = extent.tile;
    let packing = condition_packing(
        AttentionKind::Sparse {
            tile,
            sparsity,
            reference_keep,
        },
        extent.condition_rows,
    )?;
    if num_text_tokens < 1
        || num_text_tokens > extent.text_rows
        || condition_rows(segments, &packing) > extent.condition_rows
    {
        return Err(Error::config(
            "the prompt and conditions must fit their layout's capacities".to_owned(),
        ));
    }
    if vision_spans
        .iter()
        .any(|(start, stop)| !(start < stop && *stop <= num_text_tokens))
    {
        return Err(Error::config(
            "vision spans must lie within the prompt".to_owned(),
        ));
    }
    let rows = extent.padded_tokens as usize;
    let tiles = extent.padded_tokens / tile;
    let raster = latent_raster(canvas)?;
    let frames = timeline_latents(num_frames)?;
    let (spatial, width_grid) = frame_grid(raster);
    let shape = tile_shape(tile)?;

    let mut positions = vec![[0.0_f64; 3]; rows];
    let mut tags = vec![Tag::Video as i32; rows];
    let mut groups = vec![Group::Video as i32; rows];
    let mut prefix_index = vec![extent.zero_row() as i32; rows];
    let mut block_sizes = vec![0_u32; tiles as usize];
    let mut tile_segments = vec![-1_i32; tiles as usize];
    let mut tile_counts: Vec<u32> = Vec::new();

    // The prompt fills the text tiles from the first, text row i at (i, 0, 0).
    let text = dense_rows(0, num_text_tokens, tile, &mut block_sizes);
    for (offset, row) in text.clone().enumerate() {
        positions[row as usize] = [offset as f64, 0.0, 0.0];
        tags[row as usize] = Tag::Text as i32;
        prefix_index[row as usize] = offset as i32;
    }
    for (start, stop) in vision_spans {
        for row in *start..*stop {
            tags[row as usize] = Tag::Video as i32;
        }
    }

    // References advance the media clock in packed order; the conditions' prefix source rows
    // follow the text capacity. Keyframes lead and are placed on the generated timeline once the
    // references have left the clock at its origin.
    let mut clock = f64::from(num_text_tokens);
    let mut next_tile = extent.text_rows / tile;
    let mut source = extent.text_rows;
    let mut keyframes: Vec<(Segment, Range<u32>)> = Vec::new();
    for segment in segments {
        let audio = if segment.audio_rows() > 0 {
            let rows = dense_rows(next_tile, segment.audio_rows(), tile, &mut block_sizes);
            next_tile += segment.audio_rows().div_ceil(tile);
            for (offset, row) in rows.clone().enumerate() {
                prefix_index[row as usize] = (source + offset as u32) as i32;
                tags[row as usize] = Tag::Audio as i32;
                groups[row as usize] = Group::AudioCondition as i32;
            }
            source += segment.audio_rows();
            rows
        } else {
            0..0
        };
        match segment.kind {
            SegmentKind::FirstFrame | SegmentKind::LastFrame | SegmentKind::Image => {
                let visual = dense_rows(next_tile, segment.video_rows(), tile, &mut block_sizes);
                next_tile += segment.video_rows().div_ceil(tile);
                for (offset, row) in visual.clone().enumerate() {
                    prefix_index[row as usize] = (source + offset as u32) as i32;
                    tags[row as usize] = Tag::Video as i32;
                    groups[row as usize] = Group::VisualCondition as i32;
                }
                source += segment.video_rows();
                if segment.kind == SegmentKind::Image {
                    let (grid, _) = frame_grid(segment.raster);
                    for (offset, point) in grid.iter().enumerate() {
                        positions[visual.start as usize + offset] = [clock, point[0], point[1]];
                    }
                    // An image takes one integer slot, not a latent frame's span.
                    clock += 1.0;
                } else {
                    keyframes.push((*segment, visual));
                }
            }
            SegmentKind::Audio => {
                place_audio(
                    &mut positions,
                    audio,
                    segment.audio_frames,
                    clock,
                    &width_grid,
                );
                clock += f64::from(segment.audio_frames);
            }
            SegmentKind::Video => {
                let (grid, own_width) = frame_grid(segment.raster);
                if segment.audio_rows() > 0 {
                    place_audio(
                        &mut positions,
                        audio,
                        segment.audio_frames,
                        clock,
                        &own_width,
                    );
                }
                let (visual, raster_rows) = video_segment(
                    next_tile,
                    video_grid(segment.frames, segment.raster),
                    shape,
                    tile,
                    &mut block_sizes,
                    &mut tile_segments,
                    &mut tile_counts,
                );
                next_tile += tile_counts.last().copied().unwrap_or(0);
                for (row, index) in visual.iter().zip(&raster_rows) {
                    prefix_index[*row as usize] = (source + index) as i32;
                    tags[*row as usize] = Tag::Video as i32;
                    groups[*row as usize] = Group::VisualCondition as i32;
                }
                place_grid_frames(
                    &mut positions,
                    &visual,
                    &raster_rows,
                    segment.frames,
                    clock,
                    &grid,
                );
                source += segment.video_rows();
                clock += f64::from(segment.audio_frames).max(video_span(segment.frames));
            }
        }
    }

    for (segment, visual) in keyframes {
        place_keyframe(&mut positions, visual, &segment, clock, frames, &spatial);
    }

    // The generated audio and video start where the references leave the clock; their rows are the
    // layout's own.
    let audio = dense_rows(
        extent.audio_start() / tile,
        extent.audio_rows,
        tile,
        &mut block_sizes,
    );
    place_audio(
        &mut positions,
        audio.clone(),
        extent.audio_rows / AUDIO_CHANNELS,
        clock,
        &width_grid,
    );
    for row in audio {
        tags[row as usize] = Tag::Audio as i32;
        groups[row as usize] = Group::Audio as i32;
    }
    // The generated video: tile-major over its token grid, from the layout's video tiles.
    let (video_raster_indices, video_sizes) = tile_order(extent.video_grid, shape);
    let mut video_indices = Vec::with_capacity(video_raster_indices.len());
    for (index, size) in video_sizes.iter().enumerate() {
        let start = extent.video_start() + index as u32 * tile;
        video_indices.extend(start..start + size);
    }
    let video_tile = extent.video_start() / tile;
    for (offset, size) in video_sizes.iter().enumerate() {
        block_sizes[video_tile as usize + offset] = *size;
        tile_segments[video_tile as usize + offset] = tile_counts.len() as i32;
    }
    tile_counts.push(extent.video_tiles);
    place_grid_frames(
        &mut positions,
        &video_indices,
        &video_raster_indices,
        frames,
        clock,
        &spatial,
    );

    // A reference segment keeps its reference rate, the generated video its sparsity; a segment's
    // start counts every tile of a lower index, the dense and empty tiles included.
    let mut segment_keep = vec![0_i32; tiles as usize];
    let mut segment_starts = vec![0_i32; tiles as usize];
    let mut below = tile_segments.iter().filter(|index| **index < 0).count() as i32;
    for (index, count) in tile_counts.iter().enumerate() {
        let keep = if index + 1 == tile_counts.len() {
            sparsity
        } else {
            let reference_keep = reference_keep
                .ok_or_else(|| Error::config("reference videos need a keep rate".to_owned()))?;
            1.0 - reference_keep
        };
        segment_keep[index] = kept_tiles(keep, *count) as i32;
        segment_starts[index] = below;
        below += *count as i32;
    }
    Ok(RowTables {
        position_ids: positions,
        token_tags: tags,
        groups,
        prefix_index,
        used: extent.padded_tokens,
        video_indices,
        video_raster_indices,
        variable_block_sizes: block_sizes,
        tile_segments,
        segment_starts,
        segment_keep,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiles `segment` occupies in a sparse layout of `tile` rows ([`condition_tiles`]).
    fn tiles(segment: &Segment, tile: u32) -> u32 {
        let packing = condition_packing(
            AttentionKind::Sparse {
                tile,
                sparsity: 0.0,
                reference_keep: None,
            },
            0,
        )
        .expect("a tile");
        condition_tiles(segment.audio_rows(), segment.visual(), &packing)
    }

    /// The 768p 16:9 canvas: 1344x768, 84x48 latent columns and rows.
    fn canvas() -> Canvas {
        Canvas {
            width: 1344,
            height: 768,
        }
    }

    #[test]
    fn the_audio_timeline_runs_at_forty_latents_per_second() {
        for (frames, latents) in [(107, 178), (124, 207), (192, 320), (362, 603)] {
            assert_eq!(audio_latent_frames(frames), latents);
            assert_eq!(audio_rows(frames), 2 * latents);
            // One `AUDIO_HOP` of samples per audio latent step.
            assert_eq!(audio_samples(frames), latents * AUDIO_HOP);
        }
    }

    /// A request's track is a third of a hop either side of the duration it asked for, and which side
    /// is `frames % 3`.
    ///
    /// The exact duration of `n` frames is `4000n / 3` samples: a frame is not a whole number of
    /// samples at 24 fps and 32 kHz, so the comparison is stated in thirds - `3 * track` against
    /// `4000n` - and the difference is `+800`, `-800` or `0`, which is a hop's third in each
    /// direction. **The short cases are the ones a mux must not "correct":** a sound track 266.67
    /// samples shorter than its video is the model's answer for 107 and 362 frames, and padding it
    /// to the video's length is a track of zeros the checkpoint never produced.
    #[test]
    fn a_tracks_length_is_a_third_of_a_hop_from_its_request() {
        for (frames, thirds) in [(107, -800_i64), (124, 800), (192, 0), (243, 0), (362, -800)] {
            let track = i64::from(audio_samples(frames));
            assert_eq!(
                3 * track - 4000 * i64::from(frames),
                thirds,
                "{frames} frames: {} samples against an exact {}/3",
                track,
                4000 * i64::from(frames)
            );
        }
    }

    #[test]
    fn the_latent_raster_is_the_canvas_over_the_vaes_compression() {
        let raster_canvas = canvas();
        let raster = latent_raster(&raster_canvas).expect("a legal canvas");
        // The raster is the canvas over the VAE's compression, and its patch grid the raster over
        // the DiT's patch side.
        let (rows, columns) = (
            raster_canvas.height / LATENT_COMPRESSION,
            raster_canvas.width / LATENT_COMPRESSION,
        );
        assert_eq!((raster.height, raster.width), (rows, columns));
        let (patched_rows, patched_columns) = (rows / PATCH_SIDE, columns / PATCH_SIDE);
        assert_eq!(raster.patch_rows(), patched_rows * patched_columns);
        let latents = 37;
        let framed = video_grid(latents, raster);
        assert_eq!(framed, (latents, patched_rows, patched_columns));
        assert_eq!(
            framed.0 * framed.1 * framed.2,
            latents * patched_rows * patched_columns
        );

        let off_grid = Canvas {
            width: 1340,
            ..canvas()
        };
        assert!(latent_raster(&off_grid).is_err());
    }

    #[test]
    fn a_tile_order_covers_every_grid_row_once_in_whole_tiles() {
        // A grid smaller than one tile, a grid of two tiles in each axis, and a ragged grid.
        for (grid, shape) in [
            ((2, 3, 3), (4, 4, 4)),
            ((4, 8, 8), (4, 4, 4)),
            ((5, 7, 9), (4, 4, 8)),
        ] {
            let order = tile_order(grid, shape);
            let rows = (grid.0 * grid.1 * grid.2) as usize;
            assert_eq!(order.0.len(), rows, "{grid:?}");
            let mut sorted = order.0.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..rows as u32).collect::<Vec<_>>(), "{grid:?}");
            assert_eq!(order.1.iter().sum::<u32>(), rows as u32);
            assert!(
                order
                    .1
                    .iter()
                    .all(|size| *size <= shape.0 * shape.1 * shape.2),
                "{grid:?}"
            );
            // Every prefix of tiles covers exactly the rows it sizes.
            let mut below = 0;
            for taken in 1..=order.1.len() as u32 {
                below += order.1[taken as usize - 1];
                assert_eq!(
                    tiled_rows(grid, shape, taken),
                    below,
                    "{grid:?} after {taken} tiles"
                );
            }
            assert_eq!(tiled_rows(grid, shape, 0), 0);
            assert_eq!(tiled_rows(grid, shape, u32::MAX), rows as u32);
        }
    }

    #[test]
    fn a_query_keeps_at_least_one_tile_and_at_most_all_of_them() {
        assert_eq!(kept_tiles(0.8, 10), 2);
        assert_eq!(kept_tiles(0.9, 10), 1);
        assert_eq!(kept_tiles(0.0, 5), 5);
        assert_eq!(kept_tiles(0.99, 1), 1);
        assert_eq!(kept_tiles(0.5, 0), 1);
        assert_eq!(kept_tiles(0.5, 5), 3);
    }

    #[test]
    fn a_condition_occupies_whole_tiles_of_its_row_counts() {
        let raster = latent_raster(&canvas()).expect("a legal canvas");
        let keyframe = Segment {
            kind: SegmentKind::FirstFrame,
            condition: 0,
            frames: 1,
            raster,
            audio_frames: 0,
        };
        assert_eq!(keyframe.video_rows(), 1008);
        assert_eq!(keyframe.audio_rows(), 0);
        assert_eq!(tiles(&keyframe, 64), 16);
        assert_eq!(tiles(&keyframe, 128), 8);

        // A reference at a 2048-pixel short edge, with its soundtrack.
        let reference = (
            37,
            LatentRaster {
                height: 128,
                width: 72,
            },
        );
        let video = Segment {
            kind: SegmentKind::Video,
            condition: 0,
            frames: reference.0,
            raster: reference.1,
            audio_frames: 207,
        };
        assert_eq!(video.video_rows(), 37 * 64 * 36);
        assert_eq!(video.audio_rows(), 414);
        // 414 audio rows fill 7 tiles of 64; the video's 37x64x36 grid fills 10x16x9 tiles.
        assert_eq!(tiles(&video, 64), 7 + 10 * 16 * 9);

        let audio = Segment {
            kind: SegmentKind::Audio,
            condition: 1,
            frames: 0,
            raster: LatentRaster {
                height: 0,
                width: 0,
            },
            audio_frames: 207,
        };
        assert_eq!(audio.video_rows(), 0);
        assert_eq!(tiles(&audio, 128), 4);
    }

    #[test]
    fn the_image_segment_counts_one_latent_frame_whatever_its_raster() {
        let image = Segment {
            kind: SegmentKind::Image,
            condition: 0,
            frames: 1,
            raster: LatentRaster {
                height: 90,
                width: 160,
            },
            audio_frames: 0,
        };
        assert_eq!(image.video_rows(), 45 * 80);
        assert_eq!(tiles(&image, 128), (45 * 80_u32).div_ceil(128));
    }

    /// The spatial rotary grid of one axis, as the released plan defines it: evenly spaced over an
    /// interval of `extent / sqrt(area)` centred on 0.5, scaled by 32.
    fn axis_grid(extent: u32, area: f64) -> Vec<f64> {
        let ratio = f64::from(extent) / area.sqrt();
        let left = (1.0 - ratio) / 2.0;
        let count = extent as usize / 2;
        (0..count)
            .map(|index| (left + index as f64 * ratio / count as f64) * 32.0)
            .collect()
    }

    #[test]
    fn a_dense_layout_sizes_the_generated_rows_in_front_of_the_prefix() {
        let plan = dense_packing(124, &canvas(), 1935, 0, 64).expect("a dense plan");
        assert_eq!(plan.video_rows, 37 * 1008);
        assert_eq!(plan.audio_rows, 414);
        assert_eq!(plan.prefix_start(), 37 * 1008 + 414);
        assert_eq!(plan.zero_row(), 1935);
        // 39_645 rows rounded up to a whole number of 64-row alignment units.
        assert_eq!(plan.padded_tokens, 620 * 64);

        let wider = dense_packing(124, &canvas(), 1935, 2048, 64).expect("a conditioned plan");
        assert_eq!(wider.padded_tokens, 652 * 64);
        assert_eq!(wider.condition_rows, 2048);
        assert!(dense_packing(100, &canvas(), 1935, 0, 64).is_err());
        assert!(dense_packing(124, &canvas(), 0, 0, 64).is_err());
        assert!(dense_packing(124, &canvas(), 1935, 0, 0).is_err());
    }

    #[test]
    fn a_text_conditioned_dense_request_places_its_prompt_generated_rows_and_padding() {
        let plan = dense_packing(124, &canvas(), 1935, 0, 64).expect("a dense plan");
        let tables = dense_tables(&plan, 124, &canvas(), 1935, &[], &[(5, 10)])
            .expect("the released text-only request");
        assert_eq!(tables.position_ids.len(), plan.padded_tokens as usize);
        assert_eq!(tables.used, plan.prefix_start() + 1935);

        // The prompt: text row i at (i, 0, 0), rows 5..10 tagged video as vision tokens.
        for index in 0..1935u32 {
            let row = (plan.prefix_start() + index) as usize;
            assert_eq!(tables.position_ids[row], [f64::from(index), 0.0, 0.0]);
            assert_eq!(tables.prefix_index[row], index as i32);
            assert_eq!(tables.groups[row], Group::Video as i32);
            if !(5..10).contains(&index) {
                assert_eq!(tables.token_tags[row], Tag::Text as i32);
            }
        }
        for row in plan.prefix_start() + 5..plan.prefix_start() + 10 {
            assert_eq!(tables.token_tags[row as usize], Tag::Video as i32);
        }

        // The generated video starts where the prompt leaves the clock, at the target's grid.
        let (spatial, widths) = frame_grid(latent_raster(&canvas()).expect("raster"));
        assert_eq!(
            tables.position_ids[0],
            [1935.0, spatial[0][0], spatial[0][1]]
        );
        assert_eq!(tables.token_tags[0], Tag::Video as i32);
        assert_eq!(tables.groups[0], Group::Video as i32);
        assert_eq!(tables.prefix_index[0], plan.zero_row() as i32);
        let heights = axis_grid(48, f64::from(48 * 84));
        assert_eq!(spatial[0][0], heights[0]);
        assert_eq!(widths[0], axis_grid(84, f64::from(48 * 84))[0]);
        assert_eq!(*spatial.last().expect("a patch"), [heights[23], widths[41]]);

        // The generated audio is one row per channel per latent, at the ends of the width axis.
        let audio = plan.prefix_start() - plan.audio_rows;
        assert_eq!(
            tables.position_ids[audio as usize],
            [1935.0, 0.0, widths[0]]
        );
        assert_eq!(
            tables.position_ids[(audio + 206) as usize],
            [1935.0 + 206.0, 0.0, widths[0]]
        );
        assert_eq!(
            tables.position_ids[(audio + 207) as usize],
            [1935.0, 0.0, widths[41]]
        );
        for row in audio..plan.prefix_start() {
            assert_eq!(tables.token_tags[row as usize], Tag::Audio as i32);
            assert_eq!(tables.groups[row as usize], Group::Audio as i32);
        }

        // The padding after `used` is attention-excluded and reads the prefix source's zeros.
        for row in tables.used..plan.padded_tokens {
            assert_eq!(tables.position_ids[row as usize], [0.0, 0.0, 0.0]);
            assert_eq!(tables.token_tags[row as usize], Tag::Video as i32);
            assert_eq!(tables.prefix_index[row as usize], plan.zero_row() as i32);
        }
    }

    #[test]
    fn a_keyframe_sits_on_the_generated_timeline_at_the_target_grid() {
        let plan = dense_packing(124, &canvas(), 1935, 2048, 64).expect("a conditioned plan");
        let raster = latent_raster(&canvas()).expect("raster");
        let keyframe = |kind| Segment {
            kind,
            condition: 0,
            frames: 1,
            raster,
            audio_frames: 0,
        };
        let segments = [
            keyframe(SegmentKind::FirstFrame),
            keyframe(SegmentKind::LastFrame),
        ];
        let tables =
            dense_tables(&plan, 124, &canvas(), 1935, &segments, &[]).expect("two keyframes");
        let (spatial, _) = frame_grid(raster);
        assert_eq!(tables.used, plan.prefix_start() + 1935 + 2 * 1008);

        // The first keyframe shares the first generated latent frame's time ...
        let first = plan.prefix_start() + 1935;
        assert_eq!(
            tables.position_ids[first as usize],
            [1935.0, spatial[0][0], spatial[0][1]]
        );
        assert_eq!(tables.token_tags[first as usize], Tag::Video as i32);
        assert_eq!(tables.groups[first as usize], Group::VisualCondition as i32);
        // ... and the last one the last latent frame's, 205 units after the origin.
        let last = first + 1008;
        assert!(
            (tables.position_ids[last as usize][0] - 2140.0).abs() < 1e-9,
            "{}",
            tables.position_ids[last as usize][0]
        );
        assert_eq!(
            tables.position_ids[last as usize][1..],
            [spatial[0][0], spatial[0][1]]
        );
        // Conditions gather the prefix source rows after the text capacity.
        assert_eq!(tables.prefix_index[first as usize], 1935);
        assert_eq!(tables.prefix_index[last as usize], 1935 + 1008);
    }

    #[test]
    fn a_reference_request_advances_the_clock_by_an_image_and_an_audio_track() {
        let plan = dense_packing(124, &canvas(), 6913, 512, 64).expect("a dense plan");
        let image = Segment {
            kind: SegmentKind::Image,
            condition: 0,
            frames: 1,
            raster: LatentRaster {
                height: 32,
                width: 32,
            },
            audio_frames: 0,
        };
        let audio = Segment {
            kind: SegmentKind::Audio,
            condition: 1,
            frames: 0,
            raster: LatentRaster {
                height: 0,
                width: 0,
            },
            audio_frames: 124,
        };
        let tables = dense_tables(&plan, 124, &canvas(), 6913, &[image, audio], &[])
            .expect("a reference request");
        let conditions = plan.prefix_start() + 6913;
        assert_eq!(tables.used, conditions + 256 + 248);

        // The image takes one integer slot at its own 16x16 patch grid.
        let (grid, _) = frame_grid(LatentRaster {
            height: 32,
            width: 32,
        });
        assert_eq!(
            tables.position_ids[conditions as usize],
            [6913.0, grid[0][0], grid[0][1]]
        );
        assert_eq!(
            tables.groups[conditions as usize],
            Group::VisualCondition as i32
        );

        // The audio track advances the clock by its latent count, so the generated rows start at
        // 6913 + 1 + 124.
        let track = conditions + 256;
        assert_eq!(tables.groups[track as usize], Group::AudioCondition as i32);
        assert_eq!(tables.position_ids[track as usize][0], 6914.0);
        assert_eq!(tables.position_ids[0][0], 7038.0);
        assert_eq!(plan.audio_rows, 414);
        let generated_audio = plan.prefix_start() - plan.audio_rows;
        assert_eq!(tables.position_ids[generated_audio as usize][0], 7038.0);
        // An audio-only reference has no raster, so its channels read the target's width axis.
        let (_, widths) = frame_grid(latent_raster(&canvas()).expect("raster"));
        assert_eq!(
            tables.position_ids[track as usize][2], widths[0],
            "the reference track's own width axis is its 16-patch grid's"
        );
    }

    #[test]
    fn a_prompt_or_a_condition_that_does_not_fit_its_capacity_is_refused() {
        let plan = dense_packing(124, &canvas(), 1935, 0, 64).expect("a dense plan");
        let error = dense_tables(&plan, 124, &canvas(), 1936, &[], &[])
            .expect_err("the prompt exceeds the text capacity");
        // The prompt's half names the prompt's own count, so a caller reading one refusal cannot
        // mistake it for the condition half's.
        assert!(
            matches!(error, Error::Config(ref problem)
                if problem.contains("prefix segment") && problem.contains("1936 row(s)")),
            "the text half names the prompt's count: {error}"
        );

        let keyframe = Segment {
            kind: SegmentKind::FirstFrame,
            condition: 0,
            frames: 1,
            raster: latent_raster(&canvas()).expect("raster"),
            audio_frames: 0,
        };
        let error = dense_tables(&plan, 124, &canvas(), 1935, &[keyframe], &[])
            .expect_err("no condition row fits");
        // And the condition half names conditions and never the prompt - the distinction the two
        // refusals exist for.
        assert!(
            matches!(error, Error::Config(ref problem)
                if problem.contains("prefix segment")
                    && problem.contains("condition row")
                    && !problem.contains("prompt of")),
            "the condition half names conditions: {error}"
        );

        let capacity = dense_packing(124, &canvas(), 1935, 2048, 64).expect("a plan");
        let error = dense_tables(&capacity, 124, &canvas(), 1935, &[], &[(10, 5)])
            .expect_err("a vision span must be a range of the prompt");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("vision spans")),
            "{error}"
        );
    }

    #[test]
    fn a_sparse_layout_holds_whole_tiles_of_text_conditions_audio_and_video() {
        let extent = segment_extent(124, &canvas(), 7040, 512, 128, 128).expect("a sparse layout");
        // 37x24x42 grid in 4x4x8 tiles: 10x6x6 tiles.
        assert_eq!(extent.video_grid, (37, 24, 42));
        assert_eq!(extent.video_tiles, 360);
        assert_eq!(extent.audio_rows, 414);
        assert_eq!(extent.audio_start(), 7552);
        assert_eq!(extent.video_start(), 8064);
        assert_eq!(extent.zero_row(), 7552);
        // 423 whole tiles, so the alignment adds nothing.
        assert_eq!(extent.padded_tokens, 423 * 128);

        // With conditions the text tiles are the only ones every request shares.
        assert_eq!(extent.dense_prefix_tiles(), 55);
        let unconditioned = segment_extent(124, &canvas(), 7040, 0, 128, 128).expect("a layout");
        assert_eq!(unconditioned.video_start(), 7040 + 512);
        assert_eq!(unconditioned.dense_prefix_tiles(), (7040 + 512) / 128);

        // 63 dense tiles and 36 kept of the video's 360.
        let bounds = extent.key_bounds(0.9);
        assert_eq!(bounds.len(), 423);
        assert_eq!(bounds[..63], [423; 63]);
        assert_eq!(bounds[63..], [99; 360]);

        assert_eq!(extent.audio_rows_below(7552), 0);
        assert_eq!(extent.audio_rows_below(8000), 414);
        assert_eq!(extent.video_rows_below(8064).expect("a boundary"), 0);
        assert_eq!(extent.video_rows_below(8192).expect("a boundary"), 128);
        assert_eq!(extent.video_rows_below(54144).expect("a boundary"), 37296);
        assert!(extent.video_rows_below(8100).is_err());

        // Each refusal names the capacity that failed, because the five conditions point at three
        // different capacities of a deployment and a reader has to be able to tell them apart from
        // one line.
        for (text, conditions, tile, multiple, named) in [
            (7039, 512, 128, 128, "the 7039 text rows"),
            (7040, 512, 96, 96, "an H3 sparse tile holds"),
            (7040, 512, 128, 64, "the 64-row alignment"),
            (7040, 513, 128, 128, "the 513 condition rows"),
        ] {
            let error = segment_extent(124, &canvas(), text, conditions, tile, multiple)
                .expect_err("a layout off the tile grid");
            assert!(
                error.to_string().contains(named),
                "{text} {conditions} {tile} {multiple}: {error}"
            );
        }
    }

    #[test]
    fn the_generated_video_rows_of_a_sparse_layout_cover_every_patch_once() {
        let extent = segment_extent(124, &canvas(), 7040, 512, 128, 128).expect("a layout");
        let packing = segment_tables(&extent, 124, &canvas(), (1, &[]), &[], (0.9, None))
            .expect("a sparse request");
        let video_tile = (extent.video_start() / 128) as usize;
        let video_variable_block_sizes =
            &packing.variable_block_sizes[video_tile..][..extent.video_tiles as usize];
        assert_eq!(packing.video_indices.len(), 37 * 1008);
        assert_eq!(packing.video_indices[0], extent.video_start());
        assert!(
            packing
                .video_indices
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        );
        assert!(
            packing
                .video_indices
                .iter()
                .all(|row| *row < extent.video_start() + 360 * 128)
        );
        assert_eq!(video_variable_block_sizes.len(), 360);
        assert_eq!(video_variable_block_sizes.iter().sum::<u32>(), 37 * 1008);
        assert!(video_variable_block_sizes.iter().all(|size| *size <= 128));

        let mut raster = packing.video_raster_indices.clone();
        raster.sort_unstable();
        assert_eq!(raster, (0..37 * 1008).collect::<Vec<_>>());
        // Each tile's valid rows lead its block, in tile order.
        let mut taken = 0_u32;
        let mut start = extent.video_start();
        for size in video_variable_block_sizes {
            assert_eq!(packing.video_indices[taken as usize], start);
            for offset in 1..*size {
                assert_eq!(
                    packing.video_indices[(taken + offset) as usize],
                    start + offset
                );
            }
            taken += *size;
            start += 128;
        }
        assert_eq!(taken, 37 * 1008);

        assert_eq!(extent.audio_indices(), 7552..7966);
    }

    #[test]
    fn a_text_only_sparse_request_fills_its_text_audio_and_video_tiles() {
        let packing = segment_extent(124, &canvas(), 7040, 512, 128, 128).expect("a layout");
        let tables = segment_tables(
            &packing,
            124,
            &canvas(),
            (1935, &[(5, 10)]),
            &[],
            (0.9, None),
        )
        .expect("a sparse request");
        assert_eq!(tables.variable_block_sizes.len(), 423);

        // The prompt fills 15 whole tiles and 15 rows of the sixteenth.
        assert_eq!(tables.variable_block_sizes[15], 15);
        assert_eq!(tables.variable_block_sizes[16], 0);
        assert_eq!(tables.position_ids[0], [0.0, 0.0, 0.0]);
        assert_eq!(tables.position_ids[1934], [1934.0, 0.0, 0.0]);
        assert_eq!(tables.token_tags[4], Tag::Text as i32);
        assert_eq!(tables.token_tags[5], Tag::Video as i32);
        assert_eq!(tables.prefix_index[1934], 1934);

        // The generated audio fills four tiles from row 7552: 128, 128, 128 and 30.
        let audio = packing.audio_indices();
        assert_eq!(tables.variable_block_sizes[59], 128);
        assert_eq!(tables.variable_block_sizes[62], 30);
        assert_eq!(tables.variable_block_sizes[63], 128);
        let (_, widths) = frame_grid(latent_raster(&canvas()).expect("raster"));
        assert_eq!(
            tables.position_ids[audio.start as usize],
            [1935.0, 0.0, widths[0]]
        );
        assert_eq!(
            tables.position_ids[audio.start as usize + 207],
            [1935.0, 0.0, widths[41]]
        );
        assert_eq!(tables.groups[audio.start as usize], Group::Audio as i32);
        assert_eq!(tables.token_tags[audio.start as usize], Tag::Audio as i32);

        // The generated video is the last segment: 360 tiles from tile 63, keeping 36.
        assert!(tables.tile_segments[62] < 0);
        assert_eq!(tables.tile_segments[63], 0);
        assert_eq!(tables.tile_segments[422], 0);
        assert_eq!(tables.segment_starts[0], 63);
        assert_eq!(tables.segment_keep[0], 36);
        let (spatial, _) = frame_grid(latent_raster(&canvas()).expect("raster"));
        assert_eq!(
            tables.position_ids[tables.video_indices[0] as usize],
            [1935.0, spatial[0][0], spatial[0][1]]
        );
        // The generated video's tiles are the layout's own: every one holds rows, at most a tile.
        assert!(
            tables.variable_block_sizes[63..423]
                .iter()
                .all(|size| (1..=128).contains(size))
        );
    }

    #[test]
    fn a_reference_video_forms_its_own_segment_of_video_tiles() {
        let packing = segment_extent(124, &canvas(), 7040, 10240, 128, 128).expect("a layout");
        let extent = packing;
        assert_eq!(extent.video_start(), 17792);
        assert_eq!(extent.padded_tokens, 499 * 128);
        let reference = Segment {
            kind: SegmentKind::Video,
            condition: 0,
            frames: 37,
            raster: LatentRaster {
                height: 32,
                width: 32,
            },
            audio_frames: 0,
        };
        assert_eq!(tiles(&reference, 128), 80);
        let tables = segment_tables(
            &packing,
            124,
            &canvas(),
            (6913, &[]),
            std::slice::from_ref(&reference),
            (0.9, Some(0.1)),
        )
        .expect("a reference request");

        // The prompt takes 54 whole tiles and one row of the next.
        assert_eq!(tables.variable_block_sizes[53], 128);
        assert_eq!(tables.variable_block_sizes[54], 1);
        // The reference video owns tiles 55..135, each its own segment.
        assert_eq!(tables.tile_segments[55], 0);
        assert_eq!(tables.tile_segments[134], 0);
        assert!(tables.tile_segments[135] < 0);
        assert_eq!(tables.variable_block_sizes[55], 128);
        // Its rows sit at the clock the prompt leaves, at its own patch grid.
        let first = 55 * 128;
        assert_eq!(tables.position_ids[first], [6913.0, 0.0, 0.0]);
        assert_eq!(tables.groups[first], Group::VisualCondition as i32);
        assert_eq!(tables.prefix_index[first], 7040);
        assert_eq!(tables.prefix_index[first + 1], 7041);

        // The generated audio fills tiles 135..139 and the video 139..499.
        assert_eq!(tables.variable_block_sizes[135], 128);
        assert_eq!(tables.variable_block_sizes[138], 30);
        assert_eq!(tables.tile_segments[139], 1);
        // 59 dense and empty tiles below the reference, then its 80 tiles.
        assert_eq!(tables.segment_starts[0], 59);
        assert_eq!(tables.segment_keep[0], 8);
        assert_eq!(tables.segment_starts[1], 139);
        assert_eq!(tables.segment_keep[1], 36);

        // The generated rows start a reference video's span after the prompt.
        let clock = 6913.0 + 620.0 / 3.0;
        assert_eq!(tables.position_ids[extent.audio_start() as usize][0], clock);
        assert_eq!(tables.position_ids[extent.video_start() as usize][0], clock);
    }

    #[test]
    fn a_reference_request_needs_a_keep_rate_for_its_videos() {
        let packing = segment_extent(124, &canvas(), 7040, 10240, 128, 128).expect("a layout");
        let reference = Segment {
            kind: SegmentKind::Video,
            condition: 0,
            frames: 37,
            raster: LatentRaster {
                height: 32,
                width: 32,
            },
            audio_frames: 0,
        };
        let error = segment_tables(
            &packing,
            124,
            &canvas(),
            (6913, &[]),
            std::slice::from_ref(&reference),
            (0.9, None),
        )
        .expect_err("a reference video without a keep rate");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("keep rate")),
            "{error}"
        );

        // Conditions that do not fit their capacity are refused before any table exists.
        let error = segment_tables(
            &packing,
            124,
            &canvas(),
            (6913, &[]),
            &[reference; 2],
            (0.9, Some(0.1)),
        )
        .expect_err("160 tiles do not fit 80");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("capacities")),
            "{error}"
        );
    }

    /// A condition of `role` with `frames` frames of `canvas` and a track of `audio_samples`.
    fn condition(
        role: nn::ConditionRole,
        frames: u32,
        canvas: Canvas,
        audio_samples: u32,
    ) -> nn::ConditionShape {
        nn::ConditionShape {
            role,
            frames,
            height: canvas.height,
            width: canvas.width,
            audio_samples,
        }
    }

    #[test]
    fn conditions_are_ordered_keyframes_first_and_typed_by_what_they_carry() {
        let target = canvas();
        let square = Canvas {
            width: 512,
            height: 512,
        };
        let none = Canvas {
            width: 0,
            height: 0,
        };
        let conditions = [
            condition(nn::ConditionRole::Reference, 0, none, 124 * 800),
            condition(nn::ConditionRole::LastFrame, 1, target, 0),
            condition(nn::ConditionRole::Reference, 124, square, 0),
            condition(nn::ConditionRole::FirstFrame, 1, target, 0),
        ];
        let segments = condition_segments(&conditions, &target).expect("four conditions");
        // Keyframes first in request order, then the references in request order: the audio track
        // is condition 0 and the reference video condition 2.
        let kinds: Vec<_> = segments.iter().map(|segment| segment.kind).collect();
        assert_eq!(
            kinds,
            [
                SegmentKind::LastFrame,
                SegmentKind::FirstFrame,
                SegmentKind::Audio,
                SegmentKind::Video
            ]
        );
        // The keyframes keep their request order, and their condition index with it.
        assert_eq!(segments[0].condition, 1);
        assert_eq!(segments[1].condition, 3);
        assert_eq!(segments[2].condition, 0);
        assert_eq!(segments[2].audio_frames, 124);
        assert_eq!(segments[2].audio_rows(), 248);
        assert_eq!(segments[3].condition, 2);
        assert_eq!(segments[3].frames, 37);

        // A one-frame reference is an image, and an image carries no audio.
        let image = condition(nn::ConditionRole::Reference, 1, square, 0);
        let segments = condition_segments(&[image], &target).expect("an image");
        assert_eq!(segments[0].kind, SegmentKind::Image);
        assert_eq!(segments[0].frames, 1);
        let with_audio = condition(nn::ConditionRole::Reference, 1, square, 800);
        assert!(condition_segments(&[with_audio], &target).is_err());

        // A keyframe is fitted to the target canvas.
        let off_canvas = condition(nn::ConditionRole::FirstFrame, 1, square, 0);
        assert!(condition_segments(&[off_canvas], &target).is_err());
    }

    #[test]
    fn a_reference_clip_counts_by_the_encoder_s_rule_and_not_the_generated_one() {
        // The reference cuts a condition into 17-frame clips and drops the last one's tail, so a
        // 30-frame reference is 7 latent frames. The generated rule refuses that count, and rounding
        // it up to 39 frames would give 12 - five latent rows a condition does not have.
        let target = canvas();
        let count = |frames| {
            condition_segments(
                &[condition(nn::ConditionRole::Reference, frames, target, 0)],
                &target,
            )
            .expect("a referenced clip")
            .remove(0)
            .frames
        };
        assert_eq!(count(30), 7);
        assert_eq!(count(39), timeline_latents(39).expect("aligned"));
        // On an aligned count the two rules agree, which is the property that keeps them from
        // drifting: 17n + 5 frames are 5n + 2 latent frames either way.
        assert_eq!(count(124), timeline_latents(124).expect("aligned"));
        // A still is one frame and one latent frame, with no repeat to the clip count.
        assert_eq!(count(1), 1);
    }
}

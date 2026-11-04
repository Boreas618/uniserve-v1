//! The host plan of one H3 request: its frames, clips and latents, the rows its packed sequence
//! holds, and the tile order a sparse layout walks them in.
//!
//! `uniserve/models/minimax_h3/packing.py` is the released plan. Nothing here touches a device: the
//! executor uploads the tables this module computes into the batches its kernels read, so a
//! captured graph addresses them unchanged from capture to replay (specs/worker.md 4.5).
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

use crate::config::{
    AUDIO_CHANNELS, AUDIO_HOP, AttentionKind, CANVAS_MULTIPLE, CLIP_FRAMES, CLIP_LATENTS, Canvas,
    FPS, SPARSE_TILE_ROWS,
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

/// The `(height, width)` latent raster of a canvas.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LatentRaster {
    /// Latent rows: the canvas height over the VAE's 16x spatial compression.
    pub height: u32,
    /// Latent columns.
    pub width: u32,
}

impl LatentRaster {
    /// Patch rows of one latent frame: one row per 2x2 patch.
    pub fn patch_rows(&self) -> u32 {
        (self.height / 2) * (self.width / 2)
    }
}

/// The `(time, height, width)` token grid of `frames` latent frames of a latent raster.
pub fn video_grid(frames: u32, raster: LatentRaster) -> (u32, u32, u32) {
    (frames, raster.height / 2, raster.width / 2)
}

/// The latent raster of a canvas whose sides are multiples of the VAE's patch grid.
pub fn latent_raster(canvas: &Canvas) -> Result<LatentRaster> {
    if canvas.width % CANVAS_MULTIPLE != 0 || canvas.height % CANVAS_MULTIPLE != 0 {
        return Err(Error::config(format!(
            "an H3 canvas side must be a multiple of {CANVAS_MULTIPLE} pixels, got {}x{}",
            canvas.width, canvas.height
        )));
    }
    Ok(LatentRaster {
        height: canvas.height / 16,
        width: canvas.width / 16,
    })
}

/// Round a requested frame count up to the next one H3 generates: the next `17n + 5`, at least 22.
pub fn align_num_frames(requested: u32) -> u32 {
    (requested + (CLIP_FRAMES + CLIP_LATENTS - requested % CLIP_FRAMES) % CLIP_FRAMES)
        .max(CLIP_FRAMES + CLIP_LATENTS)
}

/// Latent frames a generated frame count encodes to: `5n + 2` for `n` clips.
pub fn video_latent_frames(num_frames: u32) -> Result<u32> {
    if num_frames < CLIP_FRAMES + CLIP_LATENTS || num_frames % CLIP_FRAMES != CLIP_LATENTS {
        return Err(Error::config(format!(
            "an H3 frame count has the form {CLIP_FRAMES}n + {CLIP_LATENTS}, got {num_frames}"
        )));
    }
    let clips = (num_frames - CLIP_LATENTS) / CLIP_FRAMES;
    Ok(clips * CLIP_LATENTS + 2)
}

/// Audio latent steps of a generated frame count: the duration in 40 Hz latents, rounded to the
/// nearest step.
///
/// At 24 fps and 40 Hz the exact value is never halfway between two steps, so the rounding
/// direction is unambiguous; it is round-half-to-even because the reference uses Python's `round`.
pub fn audio_latent_frames(num_frames: u32) -> u32 {
    let exact =
        f64::from(num_frames) * f64::from(crate::config::AUDIO_LATENTS_PER_SECOND) / f64::from(FPS);
    exact.round_ties_even() as u32
}

/// Audio rows of a generated frame count: one row per channel per latent step.
pub fn audio_rows(num_frames: u32) -> u32 {
    AUDIO_CHANNELS * audio_latent_frames(num_frames)
}

/// Samples of one generated frame count's soundtrack.
pub fn audio_samples(num_frames: u32) -> u32 {
    audio_latent_frames(num_frames) * AUDIO_HOP
}

/// Rows of one condition's contribution to the packed sequence.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SegmentKind {
    /// One latent frame of the target canvas anchoring a generated frame.
    Keyframe,
    /// One latent frame of a reference image at its own raster.
    Image,
    /// A reference video's latent frames.
    Video,
    /// A reference track's stereo audio rows.
    Audio,
}

/// The generated frame a keyframe anchors.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Anchor {
    /// The first generated frame.
    FirstFrame,
    /// The last generated frame.
    LastFrame,
}

/// The visual half of a condition: its latent frames and their raster.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Visual {
    /// Latent frames: one for a still, `5n + 2` for a generated-length clip.
    pub frames: u32,
    /// The latent raster of one frame.
    pub raster: LatentRaster,
}

impl Visual {
    /// Packed rows: one per 2x2 patch of every latent frame.
    pub fn rows(&self) -> u32 {
        self.frames * self.raster.patch_rows()
    }
}

/// One condition's rows in the packed sequence.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Segment {
    /// What the condition contributes.
    pub kind: SegmentKind,
    /// The condition's position in request order.
    pub condition: u32,
    /// The generated frame a keyframe anchors; `None` for every other kind.
    pub anchor: Option<Anchor>,
    /// The condition's pixels, absent for an audio-only reference.
    pub visual: Option<Visual>,
    /// Audio latent steps the condition carries.
    pub audio_frames: u32,
}

impl Segment {
    /// Visual rows: one per 2x2 patch of every latent frame.
    pub fn video_rows(&self) -> u32 {
        self.visual.map_or(0, |visual| visual.rows())
    }

    /// Audio rows: channel-major stereo.
    pub fn audio_rows(&self) -> u32 {
        AUDIO_CHANNELS * self.audio_frames
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

/// Tiles one condition occupies in a sparse layout of `tile` rows.
///
/// Its audio rows fill whole tiles, then a keyframe's or an image's rows fill whole tiles and a
/// video's rows fill the tiles of its token grid.
pub fn segment_tiles(segment: &Segment, tile: u32) -> Result<u32> {
    let shape = tile_shape(tile)?;
    let mut tiles = segment.audio_rows().div_ceil(tile);
    match segment.kind {
        SegmentKind::Keyframe | SegmentKind::Image => {
            tiles += segment.video_rows().div_ceil(tile);
        }
        SegmentKind::Video => {
            let visual = segment.visual.ok_or_else(|| {
                Error::config("an H3 video segment carries its latent frames".to_owned())
            })?;
            let grid = video_grid(visual.frames, visual.raster);
            tiles += (grid.0.div_ceil(shape.0))
                * (grid.1.div_ceil(shape.1))
                * (grid.2.div_ceil(shape.2));
        }
        SegmentKind::Audio => {}
    }
    Ok(tiles)
}

/// Tiles of a segment of `tiles` tiles that a query attends, keeping `1 - sparsity` of them.
///
/// `ceil((1 - sparsity) * tiles)`, at least one and at most every tile, in the reference's
/// double-precision arithmetic.
pub fn kept_tiles(sparsity: f32, tiles: u32) -> u32 {
    let keep = (1.0 - f64::from(sparsity)) * f64::from(tiles);
    (keep.ceil() as u32).clamp(1, tiles.max(1))
}

/// The order a `(time, height, width)` token grid is walked in whole tiles.
///
/// Tiles of `shape` cover the grid time-major, then by height, then by width; the grid's far edges
/// cut boundary tiles short. Within a tile the rows inside the grid follow in row-major order,
/// packed at the tile's front.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct TileOrder {
    /// The raster row, row-major over the grid, of every row in tile-major order.
    pub raster: Vec<u32>,
    /// The valid rows of each tile, in order: a tile's rows lead its block.
    pub sizes: Vec<u32>,
}

/// Walk `grid` in whole tiles of `shape`.
pub fn tile_order(grid: (u32, u32, u32), shape: (u32, u32, u32)) -> TileOrder {
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
    TileOrder { raster, sizes }
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

/// The raster row of each generated video row, in packed order.
///
/// Dense packing keeps the raster order; a sparse layout orders the rows tile-major, which depends
/// on neither the prompt nor the conditions.
pub fn video_order(attention: AttentionKind, num_frames: u32, canvas: &Canvas) -> Result<Vec<u32>> {
    let raster = latent_raster(canvas)?;
    let frames = video_latent_frames(num_frames)?;
    match attention {
        AttentionKind::Dense => Ok((0..frames * raster.patch_rows()).collect()),
        AttentionKind::Sparse { tile, .. } => {
            Ok(tile_order(video_grid(frames, raster), tile_shape(tile)?).raster)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 768p 16:9 canvas: 1344x768, 84x48 latent columns and rows.
    fn canvas() -> Canvas {
        Canvas {
            short_edge: 768,
            aspect: (16, 9),
            width: 1344,
            height: 768,
        }
    }

    #[test]
    fn the_frame_rule_extends_a_duration_to_the_next_whole_clip() {
        for (seconds, frames) in [(4, 107), (5, 124), (8, 192), (15, 362)] {
            assert_eq!(align_num_frames(seconds * FPS), frames, "{seconds} s");
            // A count that already has the form is its own answer.
            assert_eq!(align_num_frames(frames), frames);
        }
        // Shorter than one clip is one clip.
        assert_eq!(align_num_frames(1), 22);
        assert_eq!(align_num_frames(22), 22);
    }

    #[test]
    fn a_frame_count_encodes_to_five_latents_per_clip_and_the_trailing_overlap() {
        for (frames, latents) in [(107, 32), (124, 37), (192, 57), (362, 107)] {
            assert_eq!(
                video_latent_frames(frames).expect("a legal frame count"),
                latents
            );
            assert_eq!(latents, (frames - 5) / 17 * 5 + 2);
        }
        for illegal in [21_u32, 100, 0] {
            assert!(video_latent_frames(illegal).is_err(), "{illegal}");
        }
    }

    #[test]
    fn the_audio_timeline_runs_at_forty_latents_per_second() {
        for (frames, latents) in [(107, 178), (124, 207), (192, 320), (362, 603)] {
            assert_eq!(audio_latent_frames(frames), latents);
            assert_eq!(audio_rows(frames), 2 * latents);
            assert_eq!(audio_samples(frames), latents * 800);
        }
    }

    #[test]
    fn the_latent_raster_is_the_canvas_over_the_vaes_compression() {
        let raster = latent_raster(&canvas()).expect("a legal canvas");
        assert_eq!((raster.height, raster.width), (48, 84));
        assert_eq!(raster.patch_rows(), 24 * 42);
        let framed = video_grid(37, raster);
        assert_eq!(framed, (37, 24, 42));
        assert_eq!(framed.0 * framed.1 * framed.2, 37 * 1008);

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
            assert_eq!(order.raster.len(), rows, "{grid:?}");
            let mut sorted = order.raster.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..rows as u32).collect::<Vec<_>>(), "{grid:?}");
            assert_eq!(order.sizes.iter().sum::<u32>(), rows as u32);
            assert!(
                order
                    .sizes
                    .iter()
                    .all(|size| *size <= shape.0 * shape.1 * shape.2),
                "{grid:?}"
            );
            // Every prefix of tiles covers exactly the rows it sizes.
            let mut below = 0;
            for taken in 1..=order.sizes.len() as u32 {
                below += order.sizes[taken as usize - 1];
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
            kind: SegmentKind::Keyframe,
            condition: 0,
            anchor: Some(Anchor::FirstFrame),
            visual: Some(Visual { frames: 1, raster }),
            audio_frames: 0,
        };
        assert_eq!(keyframe.video_rows(), 1008);
        assert_eq!(keyframe.audio_rows(), 0);
        assert_eq!(segment_tiles(&keyframe, 64).expect("a tile"), 16);
        assert_eq!(segment_tiles(&keyframe, 128).expect("a tile"), 8);

        // A reference at a 2048-pixel short edge, with its soundtrack.
        let reference = Visual {
            frames: 37,
            raster: LatentRaster {
                height: 128,
                width: 72,
            },
        };
        let video = Segment {
            kind: SegmentKind::Video,
            condition: 0,
            anchor: None,
            visual: Some(reference),
            audio_frames: 207,
        };
        assert_eq!(video.video_rows(), 37 * 64 * 36);
        assert_eq!(video.audio_rows(), 414);
        // 414 audio rows fill 7 tiles of 64; the video's 37x64x36 grid fills 10x16x9 tiles.
        assert_eq!(segment_tiles(&video, 64).expect("a tile"), 7 + 10 * 16 * 9);

        let audio = Segment {
            kind: SegmentKind::Audio,
            condition: 1,
            anchor: None,
            visual: None,
            audio_frames: 207,
        };
        assert_eq!(audio.video_rows(), 0);
        assert_eq!(segment_tiles(&audio, 128).expect("a tile"), 4);
    }

    #[test]
    fn the_image_segment_counts_one_latent_frame_whatever_its_raster() {
        let image = Segment {
            kind: SegmentKind::Image,
            condition: 0,
            anchor: None,
            visual: Some(Visual {
                frames: 1,
                raster: LatentRaster {
                    height: 90,
                    width: 160,
                },
            }),
            audio_frames: 0,
        };
        assert_eq!(image.video_rows(), 45 * 80);
        assert_eq!(
            segment_tiles(&image, 128).expect("a tile"),
            (45 * 80_u32).div_ceil(128)
        );
    }

    #[test]
    fn a_dense_network_reads_the_generated_rows_in_raster_order() {
        let order = video_order(AttentionKind::Dense, 124, &canvas()).expect("a plan");
        assert_eq!(order.len(), 37 * 1008);
        assert_eq!(order[..3], [0, 1, 2]);
        assert_eq!(order[order.len() - 1], 37 * 1008 - 1);
    }

    #[test]
    fn a_sparse_network_reads_them_tile_major() {
        let order = video_order(
            AttentionKind::sparse(64, 0.8, None).expect("sparse"),
            124,
            &canvas(),
        )
        .expect("a plan");
        let tile = tile_order(
            video_grid(
                37,
                LatentRaster {
                    height: 48,
                    width: 84,
                },
            ),
            (4, 4, 4),
        );
        assert_eq!(order, tile.raster);
        // A 37x24x42 grid holds 10x6x11 tiles of 4x4x4.
        assert_eq!(tile.sizes.len(), 10 * 6 * 11);
        assert_eq!(order.len(), 37 * 1008);
    }
}

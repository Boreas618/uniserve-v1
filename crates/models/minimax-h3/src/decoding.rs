//! The H3 decoders' plan: how a video is spliced into clips, what a clip of decoded frames keeps, and
//! the layout each decoder hands the media stage.
//!
//! `uniserve/models/minimax_h3/decoding.py` and `output.py` are the reference. The video VAE decodes
//! one clip at a time and the model splices the clips into a timeline; the audio VAE decodes one
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
//! and *which* samples go where: the clip splice, the overlap the predecessor lends, and the halo a
//! decoded audio window carries.

use crate::config::VideoVaeConfig;
use crate::video_vae::{Canvas, Tile, Tiles};
use crate::{
    Error, Result,
    config::{CLIP_FRAMES, CLIP_LATENTS},
};
use nn::layers::{CrossFade, UnpatchifyVideo};
use nn::{DType, Forward, Shape, TensorView};

/// The frames one decoded clip holds: a 17-frame body, three padding frames and a five-frame tail.
pub const CLIP_OUTPUT_FRAMES: u32 = 25;

/// Frames of a clip's native output that overlap the next clip.
pub const OVERLAP_FRAMES: u32 = 5;

/// The first frame of a clip's overlap tail, within its own 25 decoded frames.
pub const TAIL_START: u32 = TAIL_START_OF(OVERLAP_FRAMES);

/// The first frame of a tail of `overlap` frames: the clip's native output less the tail.
const fn TAIL_START_OF(overlap: u32) -> u32 {
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
    if frames < CLIP_FRAMES + CLIP_LATENTS || (frames - CLIP_LATENTS) % CLIP_FRAMES != 0 {
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
    let rows = u64::from(canvas.rows()) * grid.len() as u64;
    let columns = u64::from(config.decoder_num_attention_heads)
        * u64::from(config.decoder_attention_head_dim);
    let (height, width_px) = canvas.pixels(config);
    let frames = canvas.frames_out(config);
    let pixels = grid.len() as u64 * u64::from(frames) * u64::from(height) * u64::from(width_px);
    Ok(vec![
        Buffer {
            name: "stream",
            element: "f16",
            elements: rows * columns,
        },
        Buffer {
            name: "projected",
            element: "f16",
            elements: u64::from(canvas.voxels()) * grid.len() as u64 * columns,
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
            elements: pixels * u64::from(crate::output::Rgb::CHANNELS),
        },
        Buffer {
            name: "decoded",
            element: "f16",
            elements: pixels * u64::from(crate::output::Rgb::CHANNELS),
        },
        Buffer {
            name: "rgb",
            element: "u8",
            elements: pixels * u64::from(crate::output::Rgb::CHANNELS),
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
}

impl ClipDecoder {
    /// The output path of `canvas`, whose tiles the caller decodes in [`Tile`] order.
    pub fn new(config: &VideoVaeConfig, canvas: &Canvas) -> Result<Self> {
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
        Ok(Self {
            unpatchify,
            blends,
            ratio: spatial,
            frames: canvas.frames_out(config),
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
        if tile.above != 0 {
            let previous = self.placement(decoded, tile, first_row, first_column, tile.above, 0)?;
            self.blend(tile.above, 2)?
                .forward(f, previous, view, view)?;
        }
        if tile.left != 0 {
            let previous = self.placement(decoded, tile, first_row, first_column, 0, tile.left)?;
            self.blend(tile.left, 3)?.forward(f, previous, view, view)?;
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
        latents: &[TensorView],
        positions: &[TensorView],
        tokens: TensorView,
        decoded: TensorView,
        blended: TensorView,
    ) -> Result<()> {
        if latents.len() != tiles.len() || positions.len() != tiles.len() {
            return Err(Error::config(format!(
                "{} tiles need {} latent views and {} position tables, found {} and {}",
                tiles.len(),
                tiles.len(),
                tiles.len(),
                latents.len(),
                positions.len()
            )));
        }
        let rows: u64 = tiles
            .iter()
            .map(|tile| crate::video_vae::Canvas::rows(&tile.canvas))
            .sum();
        if tokens.shape.rank() != 2 || tokens.shape.dim(0) != rows {
            return Err(Error::config(format!(
                "the clip's token buffer is {:?}, {rows} rows expected",
                tokens.shape.dims()
            )));
        }
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

    /// Blend one clip into the request's frames: its body, the tail it shares with the next clip, and the
    /// pixels the media stage reads.
    ///
    /// **Not callable yet, and deliberately so.** The splice is two operators the tree does not have:
    /// `nn::layers::CrossFade` on the frames axis (it takes rows and columns today — M1229) for the
    /// five-frame tail, whose weights are [`Clip::overlap_weights`], and `ops::rgb_bytes` for the byte
    /// raster, which also assembles the clips into one frame range because it writes through the
    /// destination's strides. Both are model's and both are asked for; until they land this function
    /// refuses rather than approximating either step on the host.
    pub fn splice(&self, clips: usize, frames: u32, height: u32, width: u32) -> Result<()> {
        let _ = (clips, frames, height, width);
        Err(Error::config(
        "the clip splice blends the frames axis and quantizes to bytes; nn::layers::CrossFade takes \
         rows and columns only (M1229) and ops::rgb_bytes is not in this tree"
            .to_owned(),
    ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn the_clip_rows_invert_the_packing_over_each_clips_seven_frames() {
        let canvas = crate::config::Canvas {
            short_edge: 768,
            aspect: (5, 3),
            width: 1280,
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
        for (index, row) in rows.iter().enumerate() {
            assert_eq!(u64::from(*row), index as u64);
        }
        // A sparse packing permutes them: every clip still reads seven consecutive latent frames,
        // which covers the whole generated video exactly once across the clips.
        let sparse = crate::config::AttentionKind::Sparse {
            tile: 128,
            sparsity: 0.8,
            reference_keep: None,
        };
        let rows = crate::packing::clip_rows(sparse, 107, &canvas).expect("rows");
        assert_eq!(rows.len() as u64, 6 * 7 * patch_rows);
        let mut seen = vec![false; (5 * 6 + 2) as usize * patch_rows as usize];
        for row in &rows {
            let index = *row as usize;
            assert!(!seen[index], "row {index} is named twice");
            seen[index] = true;
        }
        assert!(seen.iter().all(|row| *row), "a generated row is missing");
        // Clips overlap by two latent frames: the second clip starts five frames after the first.
        let per_frame = patch_rows as usize;
        assert_eq!(rows[7 * per_frame], rows[5 * per_frame]);
    }
    #[test]
    fn the_video_arena_holds_all_tiles_of_one_clip() {
        let config = crate::video_vae::tests::video_config();
        let canvas = crate::video_vae::Canvas::clip(&config, 48, 80);
        let arena = video_arena(&config, &canvas).expect("the arena");
        // 768x1280 at 256/64 is 4 x 7 tiles, each 16x16 latent voxels of 7 frames.
        assert_eq!(canvas.frames, 7);
        assert_eq!(canvas.height, 48);
        assert_eq!(canvas.width, 80);
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
        // 768x1280 pixels: 48x80 latent positions, so the checkpoint's 256-pixel tiles and 64-pixel
        // minimum overlap give a grid of four columns and seven rows, each tile 16x16 latent.
        let config = crate::video_vae::tests::video_config();
        let canvas = Canvas::clip(&config, 48, 80);
        let clip = ClipDecoder::new(&config, &canvas).expect("the output path");
        let tiles = Tiles::of(&config, &canvas)
            .expect("the grid")
            .tiles(&config, &canvas);
        assert_eq!(tiles.len(), 28);
        let frames = canvas.frames_out(&config);
        let dest = TensorView::new(
            ops::DevicePtr::new(0x1000),
            DType::F16,
            Shape::new(&[3, u64::from(frames), 768, 1280]),
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
            let origin = (first_row * 1280 + first_column) * 2;
            assert_eq!(view.ptr.address(), 0x1000 + origin);
            if tile.above != 0 {
                let previous = clip
                    .placement(dest, tile, first_row, first_column, tile.above, 0)
                    .expect("the row neighbour");
                assert_eq!(
                    previous.ptr.address(),
                    0x1000 + origin - u64::from(tile.above) * 1280 * 2
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
        // Every tile but the first row blends upwards, every tile but the first column blends left.
        assert_eq!(above_seams, 28 - 4);
        assert_eq!(left_seams, 28 - 7);
    }
}

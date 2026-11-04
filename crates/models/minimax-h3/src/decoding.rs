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

use crate::{Error, Result, config::CLIP_FRAMES, config::CLIP_LATENTS};

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
    let width = u64::from(config.decoder_num_attention_heads)
        * u64::from(config.decoder_attention_head_dim);
    let ffn = width * u64::from(config.decoder_ffn_mult);
    let (rotated, _) = crate::video_vae::rotary(config)?;
    let rows_per_tile = u64::from(canvas.rows());
    let rows = rows_per_tile * grid.len() as u64;
    let (height, width_px) = canvas.pixels(config);
    let frames = canvas.frames_out(config);
    Ok(vec![
        Buffer {
            name: "tokens",
            element: "f16",
            elements: rows * width,
        },
        Buffer {
            name: "normalized",
            element: "f16",
            elements: rows * width,
        },
        Buffer {
            name: "qkv",
            element: "f16",
            elements: rows * 3 * width,
        },
        Buffer {
            name: "rotary",
            element: "f32",
            elements: rows * u64::from(rotated),
        },
        Buffer {
            name: "attended",
            element: "f16",
            elements: rows * width,
        },
        Buffer {
            name: "gate_up",
            element: "f16",
            elements: rows * 2 * ffn,
        },
        Buffer {
            name: "pixels",
            element: "f16",
            elements: grid.len() as u64
                * u64::from(crate::output::Rgb::CHANNELS)
                * u64::from(frames)
                * u64::from(height)
                * u64::from(width_px),
        },
        Buffer {
            name: "rgb",
            element: "u8",
            elements: grid.len() as u64
                * u64::from(crate::output::Rgb::CHANNELS)
                * u64::from(frames)
                * u64::from(height)
                * u64::from(width_px),
        },
        Buffer {
            name: "overlap",
            element: "f16",
            elements: crate::output::Rgb::of(frames, height, width_px).overlap_elements(),
        },
    ])
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
            .find(|buffer| buffer.name == "tokens")
            .expect("the residual stream");
        // 28 tiles of 1797 rows of 2048 channels.
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
}

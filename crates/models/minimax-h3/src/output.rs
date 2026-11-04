//! What the H3 decoders hand the media stage: the RGB raster and the stereo PCM.
//!
//! `uniserve/models/minimax_h3/output.py` is the reference. The video decoder emits normalized RGB
//! per channel; the output stage restores it with `value * std + mean` — the ImageNet constants — then
//! clamps it to [0, 1] and quantizes it to one byte per channel. The audio decoder emits samples in
//! [-1, 1]; the output stage clamps and quantizes them to 16-bit integers. Both are the media stage's
//! input, so both layouts are stated once, here, and the decoders in [`crate::decoding`] fill them.
//!
//! ## The video raster
//!
//! `[frames, height, width, 3]` **uint8**, row-major and channel-last: the rgb24 rasters a media
//! encoder reads, with values 0 to 255. A clip is decoded to normalized frames in `[channels, frames,
//! height, width]`; the output stage denormalizes, clamps and quantizes it, and moves the channel axis
//! last as it stores.
//!
//! ## The audio raster
//!
//! `[samples, 2]` **int16**, interleaved stereo: left then right, one pair per sample. The decoder
//! writes the two channels channel-major and [`crate::audio_vae::Pcm`] states the quantization.
//!
//! ## The overlap
//!
//! Clips overlap by [`Rgb::OVERLAP_FRAMES`] frames. The decoder keeps the predecessor's tail as one
//! `[channels, 5, height, width]` block, cross-fades the successor's first five frames against it with
//! weights `i / 5`, and emits those five blended frames before the successor's body; the first blended
//! frame is entirely the predecessor's and the fifth one fifth the successor's.

use crate::{Error, Result};

/// The per-channel standard deviations the video decoder's RGB is normalized by, in red, green, blue
/// order: `value * std + mean` restores the pixel.
pub const PIXEL_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// The per-channel means the video decoder's RGB is normalized by, in red, green, blue order.
pub const PIXEL_MEAN: [f32; 3] = [0.485, 0.456, 0.406];

/// The one byte per channel the video raster stores.
pub const BYTE_MAX: f32 = 255.0;

/// The one integer step of the audio raster: a sample of `1.0` quantizes to it.
pub const PCM_MAX: f32 = 32767.0;

/// The extents of one request's video raster, and the bytes it occupies.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Rgb {
    /// Frames of the timeline.
    pub frames: u32,
    /// Pixel rows.
    pub height: u32,
    /// Pixel columns.
    pub width: u32,
}

impl Rgb {
    /// Colour channels of the raster: red, green and blue.
    pub const CHANNELS: u32 = 3;

    /// Frames one clip shares with the next.
    pub const OVERLAP_FRAMES: u32 = 5;

    /// The raster of `frames` decoded frames at a pixel extent.
    pub fn of(frames: u32, height: u32, width: u32) -> Self {
        Self {
            frames,
            height,
            width,
        }
    }

    /// The bytes the raster occupies: one per channel per pixel.
    pub fn bytes(&self) -> u64 {
        u64::from(self.frames)
            * u64::from(self.height)
            * u64::from(self.width)
            * u64::from(Self::CHANNELS)
    }

    /// The frames the decoder's own workspace holds: the longest clip, the last, which keeps its tail.
    pub const fn workspace_frames() -> u32 {
        25
    }

    /// The extent of one overlap block, `[channels, OVERLAP_FRAMES, height, width]` as the decoder
    /// keeps it between clips.
    pub fn overlap_elements(&self) -> u64 {
        u64::from(Self::CHANNELS)
            * u64::from(Self::OVERLAP_FRAMES)
            * u64::from(self.height)
            * u64::from(self.width)
    }

    /// One normalized pixel's three bytes, the reference's arithmetic.
    ///
    /// `value * std + mean` per channel, clamped to [0, 1], scaled to 255, rounded half to even, and
    /// truncated to a byte. This is the reference of the operator the output stage runs; a kernel
    /// that computes it must agree with it on every input, including the ties a half-even round
    /// resolves away from zero.
    pub fn quantize(&self, normalized: [f32; 3]) -> [u8; 3] {
        let mut bytes = [0u8; 3];
        for (channel, value) in normalized.iter().enumerate() {
            let restored = value * PIXEL_STD[channel] + PIXEL_MEAN[channel];
            let clamped = restored.clamp(0.0, 1.0);
            bytes[channel] = (clamped * BYTE_MAX).round_ties_even() as u8;
        }
        bytes
    }
}

/// The extents of one request's audio raster.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Pcm {
    /// Sample frames of the track.
    pub samples: u32,
}

impl Pcm {
    /// Interleaved channels: left and right.
    pub const CHANNELS: u32 = 2;

    /// The raster of `samples` sample frames.
    pub const fn of(samples: u32) -> Self {
        Self { samples }
    }

    /// The integers the raster holds: one per channel per sample.
    pub fn elements(&self) -> u64 {
        u64::from(self.samples) * u64::from(Self::CHANNELS)
    }

    /// The bytes the raster occupies, two per integer.
    pub fn bytes(&self) -> u64 {
        self.elements() * 2
    }

    /// One sample's integer: clamped to [-1, 1], scaled by 32767 and rounded half to even.
    ///
    /// The same arithmetic [`crate::audio_vae::Pcm::quantize`] states for the decoder's output
    /// stage; the two must agree on every input.
    pub fn quantize(&self, sample: f32) -> i16 {
        (sample.clamp(-1.0, 1.0) * PCM_MAX).round_ties_even() as i16
    }
}

/// The element type of the video raster the media stage reads.
pub const VIDEO_DTYPE: &str = "u8";

/// The element type of the audio raster the media stage reads.
pub const AUDIO_DTYPE: &str = "i16";

/// One clip's placement in the timeline, as the output stage reads it.
///
/// The body is the frames the timeline keeps from this clip; the overlap is the tail the next clip
/// cross-fades against. Both index the clip's own decoded frames, not the timeline's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Splice {
    /// The frames of the clip's own decode that the timeline keeps.
    pub body: (u32, u32),
    /// The frames of the clip's own decode that overlap the next clip.
    pub overlap: (u32, u32),
}

impl Splice {
    /// The splice the checkpoint's 25-frame clip decodes to: a 17-frame body and a 5-frame tail after
    /// three VAE padding frames.
    pub const CLIP: Self = Self {
        body: (0, 17),
        overlap: (20, 25),
    };

    /// The splice of a clip that ends the timeline: it keeps its own tail, so the last frames of the
    /// video are not lost.
    pub const LAST: Self = Self {
        body: (0, 25),
        overlap: (25, 25),
    };

    /// The frames of a clip the timeline writes, in order, when it is not last.
    pub fn kept(&self) -> (u32, u32) {
        (self.body.0, self.body.1)
    }

    /// How many frames of this clip the timeline writes.
    pub fn frames(&self) -> u32 {
        let (start, stop) = self.kept();
        stop - start
    }
}

/// The whole timeline of `clips` clips: how many frames it writes.
pub fn timeline_frames(clips: &[Splice]) -> Result<u32> {
    if clips.is_empty() {
        return Err(Error::config("a video timeline has a clip".to_owned()));
    }
    let mut frames = 0;
    for clip in clips {
        frames += clip.frames();
    }
    Ok(frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_normalized_pixel_restores_its_bytes() {
        let rgb = Rgb::of(1, 1, 1);
        // Mid grey of the normalization: value 0 restores the mean itself.
        assert_eq!(rgb.quantize([0.0, 0.0, 0.0]), [124, 116, 104]);
        // The ends clamp: a value past the range does not wrap.
        assert_eq!(rgb.quantize([4.0, 4.0, 4.0]), [255, 255, 255]);
        assert_eq!(rgb.quantize([-4.0, -4.0, -4.0]), [0, 0, 0]);
        // Every channel restores with its own constant: 1.0 is white.
        assert_eq!(rgb.quantize([1.0, 1.0, 1.0]), [255, 255, 255]);
        assert_eq!(rgb.quantize([-1.0, -1.0, -1.0]), [0, 0, 0]);
    }

    #[test]
    fn a_tie_rounds_to_even() {
        // 0.5 * 255 = 127.5 rounds to 128; 2.5 rounds to 2, not 3.
        assert_eq!(0.5f32.round_ties_even(), 0.0);
        assert_eq!(2.5f32.round_ties_even(), 2.0);
        assert_eq!(127.5f32.round_ties_even(), 128.0);
        // A pixel that lands exactly on 127.5 quantizes to 128, the even byte.
        let rgb = Rgb::of(1, 1, 1);
        let value = (127.5 / BYTE_MAX - PIXEL_MEAN[0]) / PIXEL_STD[0];
        assert_eq!(rgb.quantize([value, 0.0, 0.0])[0], 128);
    }

    #[test]
    fn a_sample_quantizes_to_the_integer_pcm() {
        let pcm = Pcm::of(800);
        assert_eq!(pcm.quantize(0.0), 0);
        assert_eq!(pcm.quantize(1.0), 32767);
        assert_eq!(pcm.quantize(-1.0), -32767);
        assert_eq!(pcm.quantize(2.0), 32767);
        assert_eq!(pcm.quantize(-2.0), -32767);
        // Half a step is a tie: it rounds to even, which here is zero.
        assert_eq!(pcm.quantize(0.5 / PCM_MAX), 0);
        assert_eq!(pcm.quantize(1.5 / PCM_MAX), 2);
        assert_eq!(pcm.elements(), 1600);
        assert_eq!(pcm.bytes(), 3200);
    }

    #[test]
    fn the_raster_sizes_follow_the_layout() {
        let rgb = Rgb::of(107, 768, 1344);
        assert_eq!(rgb.bytes(), 107 * 768 * 1344 * 3);
        assert_eq!(rgb.overlap_elements(), 3 * 5 * 768 * 1344);
        assert_eq!(Rgb::workspace_frames(), 25);
        // A clip writes 17 frames and shares 5; the last writes all 25.
        assert_eq!(Splice::CLIP.frames(), 17);
        assert_eq!(Splice::LAST.frames(), 25);
        assert_eq!(timeline_frames(&[Splice::CLIP; 6]).expect("timeline"), 102);
        let mut clips = vec![Splice::CLIP; 6];
        *clips.last_mut().expect("a clip") = Splice::LAST;
        // 5 x 17 + 25 = 110: one timeline of `17n + 5` frames with n = 6 covers 107, and the last
        // clip's tail is trimmed by the pixel extents the request states.
        assert_eq!(timeline_frames(&clips).expect("timeline"), 110);
        assert!(timeline_frames(&[]).is_err());
    }
}

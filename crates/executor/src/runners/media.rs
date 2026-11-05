//! The media group's host work: a clip encoded into its unit, a track encoded into AAC, and the
//! request's MP4 assembled from them.
//!
//! `crates/media` holds the codecs, the muxer and one assembly per request (`MediaRunner`'s
//! `MuxSession`); this module is what the executor owes it - the two published products' bytes, and
//! the one arithmetic between a decoded clip and an encoder's input. A clip is the video VAE's
//! output, `f16` NCDHW, and the encoder takes `[frames, height, width, 3]` uint8; the range that
//! maps one to the other is the model's statement (`Decoder::pixel_range`), not a number this module
//! may hold.
//!
//! Everything here is host work on blocking threads by design (`specs/worker.md` §4.5): an
//! encoding, a mux and a reader are not device computations, and the executor's `Media` lane is
//! where they run. The runner that serves them - one per deployment, with no graph, no bucket
//! capture and no device stream - is added with the mechanism hook that performs a host job.

use std::sync::Arc;

use ipc::RequestKey;

use crate::Result;

/// The rates one request's artifact is built at.
///
/// They are the model's constants, restated here as a value the executor is handed rather than
/// read: the frame rate and the clip length are the checkpoint's arithmetic
/// (`minimax_h3::config::{FPS, CLIP_FRAMES}`), the audio rate the decoder's own (`PCM_RATE`), and a
/// deployment that changed any of them would change the model's, not the muxer's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rates {
    /// Frames per second of the artifact: `24` for MiniMax-H3.
    pub frame_rate: u32,
    /// Frames one media unit covers: `17` for MiniMax-H3, the video decoder's clip length.
    pub unit_frames: u32,
    /// Samples per second of the track: `32000` for MiniMax-H3's BigVGAN.
    pub audio_rate: u32,
}

impl Rates {
    /// The frame count of every unit of a request of `frames` frames, in request order.
    ///
    /// The units are the decoder's clips, so every unit is one clip and the last one carries the
    /// remainder. A request of no frames states no unit, which the muxer refuses by name rather
    /// than assembling an empty container.
    pub fn units(&self, frames: u32) -> Vec<u32> {
        let clip = self.unit_frames.max(1);
        let mut left = frames;
        let mut units = Vec::new();
        while left > 0 {
            let unit = left.min(clip);
            units.push(unit);
            left -= unit;
        }
        units
    }

    /// The samples one request's soundtrack covers, from the same unit count.
    ///
    /// The artifact's timeline is frames at the frame rate and samples at the audio rate, so the
    /// two counts are one number read twice: a track shorter than this leaves the last frames
    /// silent, and a longer one is samples the container has no place for.
    pub fn samples(&self, units: &[u32]) -> usize {
        let frames: u64 = units.iter().map(|&frames| u64::from(frames)).sum();
        (frames * u64::from(self.audio_rate) / u64::from(self.frame_rate)) as usize
    }
}

/// One clip's `f16` NCDHW samples as the encoder's RGB24 frames, `[frames, height, width, 3]`.
///
/// `range` is the closed interval one channel's values occupy and the endpoints map to `0` and
/// `255`; a value outside it is clamped rather than wrapped, because a wrapped pixel is a colour
/// nobody computed. The clip is `[3, frames, height, width]` with red plane first, which is the
/// video VAE's output order.
///
/// # Errors
///
/// [`crate::Error::Invalid`] when the clip's length is not `3 * frames * height * width` `f16`
/// samples, when the range is not increasing, or when a sample is not finite: a non-finite value is
/// a decode that diverged, and turning it into a pixel would be the one place that could not be
/// seen.
pub fn pixels(
    clip: &[u8],
    frames: u32,
    height: u32,
    width: u32,
    range: (f32, f32),
) -> Result<Vec<u8>> {
    let (low, high) = range;
    if !(low < high) || !low.is_finite() || !high.is_finite() {
        return Err(crate::Error::Invalid(format!(
            "a clip's value range is ({low}, {high}): a range must be two finite values and \
             increasing, or every pixel of the clip is a clamped constant"
        )));
    }
    let plane = frames as usize * height as usize * width as usize;
    let want = plane * 3 * size_of::<u16>();
    if clip.len() != want {
        return Err(crate::Error::Invalid(format!(
            "a clip of {}x{}x{} frames is {want} byte(s) of f16 samples and carries {}: the \
             raster the model decoded and the raster the encoder was told of disagree",
            frames,
            height,
            width,
            clip.len()
        )));
    }
    // The scale is the range's own: the interval's ends map to the two ends of a byte.
    let scale = 255.0 / (high - low);
    let mut out = vec![0_u8; plane * 3];
    for (index, pixel) in out.iter_mut().enumerate() {
        // The output is frame-major and the clip is planar: output pixel `p`'s channel `c` is the
        // sample at `c * plane + p`. A reader that walked the planes in order instead would write
        // an image whose channels are three separate pictures.
        let sample = index % 3 * plane + index / 3;
        let bits = u16::from_ne_bytes([clip[sample * 2], clip[sample * 2 + 1]]);
        let value = half_to_f32(bits);
        if !value.is_finite() {
            return Err(crate::Error::Invalid(format!(
                "a clip's sample {sample} is {value}: a non-finite value is a decode that \
                 diverged, and no pixel states it"
            )));
        }
        let unit = ((value - low) * scale).clamp(0.0, 255.0);
        *pixel = unit.round() as u8;
    }
    Ok(out)
}

/// One little-endian `f16` bit pattern as a float.
///
/// The executor copies bytes and never converts between element types; this is the reader's
/// conversion, and it is written here rather than taken from a layer because the media group reads
/// host bytes that a device wrote in the model's own representation.
fn half_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let fraction = (bits & 0x03ff) as f32;
    match exponent {
        0 if fraction == 0.0 => sign * 0.0,
        0 => sign * fraction * 2.0_f32.powi(-24),
        31 if fraction == 0.0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + fraction / 1024.0) * 2.0_f32.powi(exponent - 15),
    }
}

/// One request's media output: the encodes of its clips and track, and the MP4 they assemble.
///
/// The assembly itself lives in `crates/media`'s `MediaRunner`, one `MuxSession` per request, so
/// this value holds no per-request state of its own and a request that ends without an artifact
/// leaves nothing here behind. Calls for one request are serialised by the mechanism: a request's
/// media calls are its own graph's and run in its order.
pub struct Output {
    media: Arc<media::MediaRunner>,
    rates: Rates,
    /// The requests whose assembly this output opened. The media crate refuses a second open for
    /// one request, which is the rule that keeps two units of a request from starting two
    /// containers, so the first clip of a request opens and every later one appends.
    opened: std::sync::Mutex<std::collections::HashSet<u64>>,
}

impl Output {
    /// An output over `media`, building every artifact at `rates`.
    pub fn new(media: Arc<media::MediaRunner>, rates: Rates) -> Self {
        Self {
            media,
            rates,
            opened: std::sync::Mutex::new(std::collections::HashSet::new()),
        }
    }

    /// The rates every artifact of this output is built at.
    pub fn rates(&self) -> Rates {
        self.rates
    }

    /// Encodes one clip of `request` into the unit it covers.
    ///
    /// `frames` is the request's whole frame count, which fixes the timeline the assembly covers;
    /// the first call opens it. `clip` is the product's bytes and `shape` the decoded raster the
    /// fragment covers - `(frames, height, width)` of this unit, taken from the clip the decoder
    /// published rather than recomputed here.
    pub fn encode_clip(
        &self,
        request: RequestKey,
        frames: u32,
        shape: (u32, u32, u32),
        clip: &[u8],
        range: (f32, f32),
    ) -> Result<()> {
        let (unit_frames, height, width) = shape;
        let units = self.rates.units(frames);
        if !units.contains(&unit_frames) {
            return Err(crate::Error::Invalid(format!(
                "a clip of {unit_frames} frames is not one of the request's units {units:?}: the \
                 decoder's clips and the assembly's timeline are the same list"
            )));
        }
        let opening = {
            let mut opened = self
                .opened
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            opened.insert(request_id(request))
        };
        if opening {
            self.media.open_mux(
                request_id(request),
                width,
                height,
                self.rates.frame_rate,
                self.rates.audio_rate,
                &units,
            )?;
        }
        let rgb24 = pixels(clip, unit_frames, height, width, range)?;
        let row = self.rates.encoded_unit_bytes(unit_frames, height, width);
        let mut framed = vec![0_u8; row];
        let written = self.media.encode_unit(
            media::Rgb24Frames {
                pixels: &rgb24,
                height,
                width,
            },
            self.rates.frame_rate,
            &mut framed,
        )?;
        framed.truncate(written);
        self.media.append_units(request_id(request), &[framed])?;
        Ok(())
    }

    /// Encodes `request`'s `[samples, 2]` track into its assembly.
    pub fn encode_track(&self, request: RequestKey, pcm: &[u8], samples: usize) -> Result<()> {
        if pcm.len() != samples * 2 * size_of::<i16>() {
            return Err(crate::Error::Invalid(format!(
                "a track of {samples} stereo samples is {} byte(s) and carries {}: the decoded \
                 window and the bytes handed to the encoder disagree",
                samples * 2 * size_of::<i16>(),
                pcm.len()
            )));
        }
        let pcm: Vec<i16> = pcm
            .chunks_exact(2)
            .map(|pair| i16::from_ne_bytes([pair[0], pair[1]]))
            .collect();
        let track = self
            .media
            .encode_audio(&pcm, samples, self.rates.audio_rate)?;
        self.media.set_audio(request_id(request), track)?;
        Ok(())
    }

    /// Finishes `request`'s assembly and answers its MP4 bytes.
    pub fn finish(&self, request: RequestKey) -> Result<Vec<u8>> {
        self.opened
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(&request_id(request));
        Ok(self.media.finish_mux(request_id(request))?)
    }

    /// Drops `request`'s assembly, for a request that ends before it could be finished.
    pub fn discard(&self, request: RequestKey) {
        self.opened
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(&request_id(request));
        self.media.close_mux(request_id(request));
    }
}

impl Rates {
    /// Bytes of one encoded unit's framed row at this raster.
    fn encoded_unit_bytes(&self, frames: u32, height: u32, width: u32) -> usize {
        media::encoded_unit_bytes(frames.min(self.unit_frames), height, width)
    }
}

/// A request's identity for the media crate, whose assembly is keyed by a number.
///
/// One deployment serves one epoch's requests, and the key already carries the epoch: the media
/// crate needs only the number within it.
fn request_id(request: RequestKey) -> u64 {
    request.request_id.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The units of a request are its decoder's clips, and the last one carries the remainder.
    ///
    /// This is the timeline every artifact is checked against: a unit count that disagreed with the
    /// clips the decoder publishes would mux a request with frames missing or with a unit the
    /// encoder never made.
    #[test]
    fn the_units_of_a_request_are_its_clips() {
        let rates = Rates {
            frame_rate: 24,
            unit_frames: 17,
            audio_rate: 32_000,
        };
        assert_eq!(rates.units(17), vec![17]);
        assert_eq!(rates.units(34), vec![17, 17]);
        assert_eq!(rates.units(96), vec![17, 17, 17, 17, 17, 11]);
        assert_eq!(rates.units(0), Vec::<u32>::new());
        // The soundtrack covers every frame of every unit at the artifact's rate.
        assert_eq!(rates.samples(&rates.units(96)), 96 * 32_000 / 24);
    }

    /// One clip becomes RGB24 in the model's stated range, in the encoder's frame-major order.
    ///
    /// The three planes are red, green and blue; the range's endpoints are the byte's ends and a
    /// value outside it is clamped. A reader that wrote the channels in the wrong order would pass
    /// every count and produce a clip whose red and blue are swapped.
    #[test]
    fn a_clip_becomes_the_encoders_frames_in_its_stated_range() {
        let planes: [[f32; 4]; 3] = [
            [-1.0, 0.0, 1.0, 2.0],
            [0.5, 0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.0, 1.0],
        ];
        let clip: Vec<u8> = planes
            .into_iter()
            .flatten()
            .map(f32_to_half)
            .flat_map(u16::to_ne_bytes)
            .collect();
        let rgb = pixels(&clip, 1, 2, 2, (-1.0, 1.0)).expect("one clip of the stated range");
        assert_eq!(rgb.len(), 12);
        // The four pixels, in the encoder's order: red, green, blue of each.
        let pixels: [[u8; 3]; 4] = [
            [0, 191, 64],    // red at the low end, green mid, blue between
            [128, 191, 191], // red at zero is the middle of the range
            [255, 191, 128], // red at the high end is the last byte
            [255, 191, 255], // red above it clamps rather than wrapping
        ];
        for (index, want) in pixels.iter().enumerate() {
            assert_eq!(&rgb[index * 3..index * 3 + 3], want, "pixel {index}");
        }
    }

    /// A clip whose byte count disagrees with the raster is refused rather than read past.
    #[test]
    fn a_clip_that_disagrees_with_its_raster_is_refused() {
        let error = pixels(&[0_u8; 8], 1, 2, 2, (-1.0, 1.0)).expect_err("a short clip");
        let message = format!("{error}");
        assert!(
            message.contains("24"),
            "the refusal names the raster: {message}"
        );
    }

    /// The range is the model's, and a range that is not a range is refused.
    #[test]
    fn a_range_that_is_not_increasing_is_refused() {
        let clip = vec![0_u8; 3 * 4 * 2];
        for range in [(1.0, 1.0), (1.0, -1.0), (f32::NAN, 1.0)] {
            assert!(pixels(&clip, 2, 1, 1, range).is_err(), "{range:?}");
        }
    }

    /// A non-finite sample is a diverged decode, and it never becomes a pixel.
    #[test]
    fn a_non_finite_sample_is_refused() {
        let mut clip: Vec<u8> = [0.0_f32; 12]
            .iter()
            .map(|&value| f32_to_half(value))
            .flat_map(u16::to_ne_bytes)
            .collect();
        clip[0..2].copy_from_slice(&f32_to_half(f32::NAN).to_ne_bytes());
        let error = pixels(&clip, 1, 2, 2, (-1.0, 1.0)).expect_err("a NaN sample");
        assert!(format!("{error}").contains("non-finite"), "{error}");
    }

    /// `f16` bits for a test's sample, written out rather than taken from a kernel: the reader's
    /// conversion is what is under test, and a fixture that used the code under test would agree
    /// with itself.
    fn f32_to_half(value: f32) -> u16 {
        let bits = value.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        let mantissa = ((bits >> 13) & 0x03ff) as u16;
        if value == 0.0 {
            return sign;
        }
        if value.is_nan() {
            return sign | 0x7c00 | 0x0200;
        }
        if exponent <= 0 {
            return sign;
        }
        if exponent >= 31 {
            return sign | 0x7c00;
        }
        sign | ((exponent as u16) << 10) | mantissa
    }
}

//! Output codecs: one media unit's H.264 stream and one request's AAC track.
//!
//! A media unit is `frames` RGB24 frames of `height` x `width`, encoded by a
//! fresh libx264 session so that its first frame is a keyframe, then muxed
//! into a self-contained MP4. The soundtrack is one AAC MP4 over the video's
//! duration. Both are the serving options of `uniserve/media/codec.py`.

/// The video codec every served container uses.
pub const VIDEO_CODEC: &str = "libx264";

/// The audio codec every served container uses.
pub const AUDIO_CODEC: &str = "aac";

/// Threads one serving encoder session uses.
///
/// Several ranks encode at once on one host, so a session must not size its
/// thread pool to the machine. Eight threads encode a unit as fast as an
/// unbounded pool at this preset, and the worker's bound on an encoded unit's
/// bytes budgets one slice header per thread per frame, so it follows this
/// value.
pub const ENCODER_THREADS: u32 = 8;

/// Samples of one planar AAC frame.
pub const AUDIO_FRAME_SAMPLES: usize = 1024;

/// A borrowed sequence of RGB24 frames, `[frames, height, width, 3]` uint8,
/// frame-major and contiguous.
#[derive(Clone, Copy, Debug)]
pub struct Rgb24Frames<'a> {
    /// `frames * height * width * 3` bytes.
    pub pixels: &'a [u8],
    /// Height of one frame.
    pub height: u32,
    /// Width of one frame.
    pub width: u32,
}

impl Rgb24Frames<'_> {
    /// Bytes of one frame.
    pub fn frame_bytes(&self) -> usize {
        self.height as usize * self.width as usize * 3
    }

    /// Number of frames the pixels hold.
    pub fn frames(&self) -> usize {
        self.pixels
            .len()
            .checked_div(self.frame_bytes())
            .unwrap_or(0)
    }
}

/// Verifies that the linked FFmpeg runtime provides the serving encoders.
///
/// # Errors
///
/// [`crate::Error::MissingCodec`] naming a serving encoder the runtime does
/// not provide, [`crate::Error::Codec`] when a provided encoder cannot open.
pub fn require_codecs() -> crate::Result<()> {
    crate::ffi::require_codecs()
}

/// Encodes one media unit into a self-contained MP4 starting at a keyframe.
///
/// Frame timestamps start at zero in every unit; the container assembly shifts
/// them onto the request's timeline ([`crate::mux::Container::append`]).
///
/// # Errors
///
/// [`crate::Error::Plan`] when the frames hold no frame, are not exactly
/// `height * width * 3` bytes per frame, or the frame rate is not positive;
/// [`crate::Error::Codec`] when the encoder or muxer fails.
pub fn encode_video_unit(frames: Rgb24Frames<'_>, frame_rate: u32) -> crate::Result<Vec<u8>> {
    let count = frames.frames();
    if frames.height == 0 || frames.width == 0 || count < 1 {
        return Err(crate::Error::plan(
            "video capture has invalid RGB24 dimensions",
        ));
    }
    if frames.pixels.len() != count * frames.frame_bytes()
        || count > i32::MAX as usize
        || frames.width > i32::MAX as u32
        || frames.height > i32::MAX as u32
    {
        return Err(crate::Error::plan(
            "video capture has invalid RGB24 dimensions",
        ));
    }
    if frame_rate < 1 || frame_rate > i32::MAX as u32 {
        return Err(crate::Error::plan(
            "video frame rate must be a positive integer",
        ));
    }
    let count = count as u32;
    crate::ffi::encode_video_unit(
        frames.pixels,
        count,
        frames.height,
        frames.width,
        frame_rate,
    )
}

/// Encodes interleaved int16 stereo PCM as a self-contained AAC MP4.
///
/// The track is truncated or zero-padded to `samples` (the video's duration at
/// `rate`), encoded in planar frames of [`AUDIO_FRAME_SAMPLES`] samples, the
/// last one zero-padded when it does not fill.
///
/// # Errors
///
/// [`crate::Error::Plan`] when `pcm` is not a whole number of stereo pairs,
/// `samples` or `rate` are not positive; [`crate::Error::Codec`] when the
/// encoder or muxer fails.
pub fn encode_audio_track(pcm: &[i16], samples: usize, rate: u32) -> crate::Result<Vec<u8>> {
    if !pcm.len().is_multiple_of(2) {
        return Err(crate::Error::plan(
            "audio capture has invalid stereo dimensions",
        ));
    }
    if samples < 1 || rate < 1 || samples > i32::MAX as usize || rate > i32::MAX as u32 {
        return Err(crate::Error::plan(
            "audio track length and rate must be positive",
        ));
    }
    // The soundtrack is the video's duration: samples past the captured track
    // are silence, and captured samples past that duration are dropped.
    let available = pcm.len() / 2;
    if available >= samples {
        crate::ffi::encode_audio_track(&pcm[..samples * 2], samples as u32, rate)
    } else {
        let mut padded = Vec::with_capacity(samples * 2);
        padded.extend_from_slice(pcm);
        padded.resize(samples * 2, 0);
        crate::ffi::encode_audio_track(&padded, samples as u32, rate)
    }
}

/// An upper bound on the MP4 bytes of one media unit.
///
/// The serving encoder is libx264, yuv420p, `preset=ultrafast`,
/// `tune=zerolatency`. That preset uses baseline CAVLC. A padded 16x16
/// macroblock has at most 27 residual blocks (including the separate DC
/// blocks), each with up to 16 coefficients. Baseline level codes occupy at
/// most 28 bits; x264 re-encodes CAVLC overflows at a higher QP. Per-block
/// token, zero and run codes need at most 16 + 9 + 15 * 11 bits. A further
/// 1536 bits covers all macroblock headers and 16 pairs of motion-vector
/// differences. NAL escaping adds at most one byte per two source bytes.
/// Each encoder thread writes one slice header per frame of at most 1024
/// bytes; MP4 sample tables need at most 64 bytes per packet, and parameter
/// sets, encoder SEI and fixed container boxes fit in 64 KiB. These are syntax
/// bounds, independent of image entropy (x264 `encoder/cavlc.c`,
/// `encoder/encoder.c` and FFmpeg `libavformat/movenc.c`).
pub fn encoded_video_bytes(frames: u32, height: u32, width: u32) -> usize {
    let blocks = height.div_ceil(16) as usize * width.div_ceil(16) as usize;
    let residual_bits: usize = 27 * (16 + 16 * 28 + 9 + 15 * 11);
    let macroblock_bytes = (residual_bits + 1536).div_ceil(8);
    let escaped_bytes = (3 * macroblock_bytes).div_ceil(2);
    (1 << 16) + frames as usize * (blocks * escaped_bytes + 8 * 1024 + 64)
}

/// Bytes of one `encoded_units` row: the length prefix and the largest
/// encoding of a unit of `frames` frames at the raster.
pub fn encoded_unit_bytes(frames: u32, height: u32, width: u32) -> usize {
    size_of::<u64>() + encoded_video_bytes(frames, height, width)
}

/// Writes `payload` behind its native-endian `u64` length into `row`.
///
/// Returns the framed length; bytes past it stay untouched, so only that
/// prefix of the row is published.
///
/// # Errors
///
/// [`crate::Error::Plan`] when the payload does not fit the row.
pub fn frame_unit(payload: &[u8], row: &mut [u8]) -> crate::Result<usize> {
    let capacity = row.len().saturating_sub(size_of::<u64>());
    if row.len() < size_of::<u64>() || payload.len() > capacity {
        return Err(crate::Error::plan(format!(
            "encoded media unit of {} bytes exceeds the {} bytes reserved for it",
            payload.len(),
            capacity
        )));
    }
    let framed = size_of::<u64>() + payload.len();
    row[..size_of::<u64>()].copy_from_slice(&(payload.len() as u64).to_ne_bytes());
    row[size_of::<u64>()..framed].copy_from_slice(payload);
    Ok(framed)
}

/// The payload of one framed unit row.
///
/// # Errors
///
/// [`crate::Error::Plan`] when the row has no length prefix or names an
/// invalid length.
pub fn read_unit(row: &[u8]) -> crate::Result<&[u8]> {
    let prefix = row
        .get(..size_of::<u64>())
        .ok_or_else(|| crate::Error::plan("encoded media unit has no length prefix"))?;
    let Ok(prefix) = <[u8; size_of::<u64>()]>::try_from(prefix) else {
        return Err(crate::Error::plan(
            "encoded media unit has no length prefix",
        ));
    };
    let length = u64::from_ne_bytes(prefix);
    let length = usize::try_from(length)
        .map_err(|_| crate::Error::plan("encoded media unit names an invalid length"))?;
    let body = &row[size_of::<u64>()..];
    if length > body.len() {
        return Err(crate::Error::plan(
            "encoded media unit names an invalid length",
        ));
    }
    Ok(&body[..length])
}

/// Samples of the soundtrack at the audio rate: the video's duration,
/// `frames * audio_rate / frame_rate`, rounded half to even.
pub fn audio_samples(frames: usize, frame_rate: u32, audio_rate: u32) -> usize {
    let numerator = frames as u128 * audio_rate as u128;
    let denominator = frame_rate as u128;
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    let twice = 2 * remainder;
    let rounded = if twice > denominator || (twice == denominator && quotient % 2 == 1) {
        quotient + 1
    } else {
        quotient
    };
    // A count this large cannot arise from a served request; saturating keeps
    // the arithmetic total.
    usize::try_from(rounded).unwrap_or(usize::MAX)
}

#[cfg(test)]
mod tests {
    use super::{
        Rgb24Frames, audio_samples, encode_audio_track, encode_video_unit, encoded_unit_bytes,
        encoded_video_bytes, frame_unit, read_unit,
    };
    use crate::Error;

    #[test]
    fn a_media_unit_is_bounded_from_its_frames_and_raster() {
        // The documented syntax bound, term for term.
        let (frames, height, width) = (17u32, 48u32, 64u32);
        let blocks = height.div_ceil(16) as usize * width.div_ceil(16) as usize;
        let macroblock_bytes = (27usize * (16 + 16 * 28 + 9 + 15 * 11) + 1536).div_ceil(8);
        let escaped_bytes = (3 * macroblock_bytes).div_ceil(2);
        let expected = (1 << 16) + frames as usize * (blocks * escaped_bytes + 8 * 1024 + 64);
        assert_eq!(encoded_video_bytes(frames, height, width), expected);
        assert_eq!(
            encoded_unit_bytes(frames, height, width),
            expected + size_of::<u64>()
        );

        // More frames or a larger raster never shrink the bound.
        assert!(encoded_video_bytes(frames + 1, height, width) > expected);
        assert!(encoded_video_bytes(frames, height * 2, width * 2) > expected);
    }

    #[test]
    fn a_framed_unit_round_trips_through_its_row() {
        let payload = b"an encoded unit";
        let mut row = vec![0xa5u8; payload.len() + size_of::<u64>() + 8];
        let framed = frame_unit(payload, &mut row).expect("the payload fits");
        assert_eq!(framed, payload.len() + size_of::<u64>());
        // The length prefix is native endian, and the row past the framed
        // length is not published.
        assert_eq!(
            row[..size_of::<u64>()],
            (payload.len() as u64).to_ne_bytes()
        );
        assert_eq!(&row[8..framed], payload);
        assert!(row[framed..].iter().all(|byte| *byte == 0xa5));
        assert_eq!(read_unit(&row[..framed]).expect("a framed row"), payload);
        // A row longer than the frame carries only the framed bytes.
        assert_eq!(read_unit(&row).expect("a framed row"), payload);
    }

    #[test]
    fn a_unit_that_does_not_fit_its_row_is_refused() {
        let mut row = vec![0u8; size_of::<u64>() + 4];
        assert!(matches!(
            frame_unit(&[0u8; 5], &mut row),
            Err(Error::Plan(_))
        ));
        assert!(frame_unit(&[0u8; 4], &mut row).is_ok());
        let mut short = vec![0u8; size_of::<u64>() - 1];
        assert!(matches!(frame_unit(&[], &mut short), Err(Error::Plan(_))));
    }

    #[test]
    fn a_row_without_a_valid_length_is_refused() {
        assert!(matches!(read_unit(&[0u8; 3]), Err(Error::Plan(_))));
        let mut row = 9u64.to_ne_bytes().to_vec();
        row.extend_from_slice(&[0u8; 4]);
        assert!(matches!(read_unit(&row), Err(Error::Plan(_))));
    }

    #[test]
    fn a_soundtrack_is_the_video_duration_rounded_half_to_even() {
        // 39 frames at 24 fps are 1.625 s: 52 000 samples at 32 kHz.
        assert_eq!(audio_samples(39, 24, 32_000), 52_000);
        assert_eq!(audio_samples(24, 24, 48_000), 48_000);
        // Half a sample rounds to the even count.
        assert_eq!(audio_samples(1, 2, 1), 0);
        assert_eq!(audio_samples(3, 2, 1), 2);
        assert_eq!(audio_samples(1, 3, 1), 0);
        assert_eq!(audio_samples(2, 3, 1), 1);
    }

    #[test]
    fn malformed_captures_are_refused_before_a_codec_runs() {
        let empty = Rgb24Frames {
            pixels: &[],
            height: 8,
            width: 8,
        };
        assert_eq!(empty.frames(), 0);
        assert!(matches!(encode_video_unit(empty, 24), Err(Error::Plan(_))));

        let partial = Rgb24Frames {
            pixels: &[0u8; 10],
            height: 8,
            width: 8,
        };
        assert!(matches!(
            encode_video_unit(partial, 24),
            Err(Error::Plan(_))
        ));

        let frame = vec![0u8; 8 * 8 * 3];
        let unit = Rgb24Frames {
            pixels: &frame,
            height: 8,
            width: 8,
        };
        assert_eq!(unit.frames(), 1);
        assert!(matches!(encode_video_unit(unit, 0), Err(Error::Plan(_))));

        assert!(matches!(
            encode_audio_track(&[0i16; 3], 1, 32_000),
            Err(Error::Plan(_))
        ));
        assert!(matches!(
            encode_audio_track(&[0i16; 2], 0, 32_000),
            Err(Error::Plan(_))
        ));
        assert!(matches!(
            encode_audio_track(&[0i16; 2], 1, 0),
            Err(Error::Plan(_))
        ));
    }
}

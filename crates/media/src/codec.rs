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
pub fn require_codecs() -> crate::Result<()> {
    todo!("builder01: encode module")
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
    todo!("builder01: encode module")
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
    todo!("builder01: encode module")
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
    todo!("builder01: encode module")
}

/// The payload of one framed unit row.
///
/// # Errors
///
/// [`crate::Error::Plan`] when the row has no length prefix or names an
/// invalid length.
pub fn read_unit(row: &[u8]) -> crate::Result<&[u8]> {
    todo!("builder01: encode module")
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

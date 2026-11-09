//! The probed facts of a request's media, which a model plans its conditions from.

/// The probed facts of one condition's media.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaMetadata {
    /// An image.
    Image(ImageMetadata),
    /// A video and its soundtrack.
    Video(VideoMetadata),
    /// An audio file.
    Audio(AudioMetadata),
}

/// An image as displayed, after its EXIF orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageMetadata {
    /// Displayed width in pixels.
    pub width: u32,
    /// Displayed height in pixels.
    pub height: u32,
}

/// A rational frame rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameRate {
    /// Frames per `denominator` seconds.
    pub numerator: u32,
    /// Seconds per `numerator` frames.
    pub denominator: u32,
}

/// A video's display aspect and size, frame rate, frame count and
/// soundtrack.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoMetadata {
    /// Width of the display aspect, in lowest terms. Only the ratio to
    /// `display_height` matters.
    pub display_width: u64,
    /// Height of the display aspect, in lowest terms.
    pub display_height: u64,
    /// Displayed width in pixels: the coded width scaled by the sample
    /// aspect ratio, rounded half to even, and turned with `height` by the
    /// display rotation.
    pub width: u32,
    /// Displayed height in pixels.
    pub height: u32,
    /// The stream's average frame rate.
    pub frame_rate: FrameRate,
    /// Frames a decoder outputs for the whole stream.
    pub frames: u64,
    /// The first audio stream, when the video has one.
    pub soundtrack: Option<AudioMetadata>,
}

/// A soundtrack's native sample rate and decoded sample count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioMetadata {
    /// Samples per second.
    pub sample_rate: u32,
    /// Samples per channel a decoder outputs.
    pub samples: u64,
}

/// Samples a track of `samples` samples at `from` Hz holds once resampled to `to` Hz.
///
/// This is torchaudio's resampler's length: `ceil(samples * new / orig)` over the gcd-reduced
/// rates `new / orig`, so a track at the target rate keeps its length and any other is rounded up.
pub const fn resampled_len(samples: u64, from: u32, to: u32) -> u64 {
    if from == to {
        return samples;
    }
    let divisor = gcd(from as u64, to as u64);
    let new = to as u64 / divisor;
    let orig = from as u64 / divisor;
    (samples * new).div_ceil(orig)
}

/// The greatest common divisor of `a` and `b`.
const fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_track_resamples_to_the_ceiling_of_its_reduced_rates() {
        assert_eq!(resampled_len(48_000, 32_000, 32_000), 48_000);
        // 44.1 kHz to 32 kHz is 320 / 441: 1000 samples are 725.6, rounded up.
        assert_eq!(resampled_len(1_000, 44_100, 32_000), 726);
        assert_eq!(resampled_len(3, 48_000, 32_000), 2);
    }
}

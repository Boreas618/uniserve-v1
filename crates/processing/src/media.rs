//! The probed facts of a request's media, which a model prepares its conditions from: the
//! processing interface's types (`scheduling::processing::video`), and the length of a resampled
//! track.

pub use scheduling::processing::video::{
    AudioMetadata, FrameRate, ImageMetadata, MediaMetadata, VideoMetadata,
};

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

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

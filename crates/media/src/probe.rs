//! Media probing: what a condition's media is, read by the decoders that read
//! it ([`crate::read`]), so the extent a request is planned with is the extent
//! the reader produces.
//!
//! - [`probe_image`]: an image's format and its displayed size, its EXIF
//!   orientation applied by the `image` decoder [`crate::read::read_image`]
//!   decodes it with (orientations 5 to 8 swap the sides).
//! - [`probe_container`]: an audio or video file's demuxer and streams, read
//!   by libavformat through the demuxers the caller allows, without decoding.
//!   A video stream's frame count is the number of packets a decoder outputs
//!   (packets an edit list discards excluded).
//! - [`decoded_track`]: an audio track's rate, channels and samples as
//!   [`crate::read::read_audio`] decodes them.
//!
//! What a request accepts (formats, codecs, rates, channel layouts) is the
//! caller's rule; this module only reports.

use std::time::Duration;

use crate::{Error, Result, ffi};

/// An image as its decoder reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageFacts {
    /// The encoded format.
    pub format: image::ImageFormat,
    /// The displayed width, orientation applied.
    pub width: u32,
    /// The displayed height, orientation applied.
    pub height: u32,
}

/// Reads an image's format and displayed size from its header, without
/// decoding its pixels.
///
/// # Errors
///
/// [`Error::Codec`] when the format is unknown or the header cannot be read.
pub fn probe_image(encoded: &[u8]) -> Result<ImageFacts> {
    use image::ImageDecoder as _;
    use image::metadata::Orientation;

    let reader = image::ImageReader::new(std::io::Cursor::new(encoded))
        .with_guessed_format()
        .map_err(|error| Error::Codec(format!("reading an image's format: {error}")))?;
    let format = reader
        .format()
        .ok_or_else(|| Error::Codec("the image's format is not known".to_owned()))?;
    let mut decoder = reader
        .into_decoder()
        .map_err(|error| Error::Codec(format!("reading an image's header: {error}")))?;
    let (width, height) = decoder.dimensions();
    // The reader applies the same orientation; a missing or unknown tag leaves
    // the image as it is.
    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    let transposed = matches!(
        orientation,
        Orientation::Rotate90
            | Orientation::Rotate270
            | Orientation::Rotate90FlipH
            | Orientation::Rotate270FlipH
    );
    Ok(if transposed {
        ImageFacts {
            format,
            width: height,
            height: width,
        }
    } else {
        ImageFacts {
            format,
            width,
            height,
        }
    })
}

/// An audio or video file's demuxer and streams.
#[derive(Clone, Debug, PartialEq)]
pub struct Container {
    /// The demuxer's names, as FFmpeg states them (`mov,mp4,m4a,3gp,3g2,mj2`).
    pub format_name: String,
    /// The first video stream that is not cover art.
    pub video: Option<VideoStream>,
    /// The first audio stream.
    pub audio: Option<AudioStream>,
}

impl Container {
    /// Whether one of the demuxer's names is `name`.
    pub fn is_format(&self, name: &str) -> bool {
        self.format_name
            .split(',')
            .any(|format| format.trim() == name)
    }
}

/// A video stream, as its container declares it.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoStream {
    /// The codec, as FFmpeg names it (`h264`, `hevc`).
    pub codec: String,
    /// The coded width.
    pub width: u32,
    /// The coded height.
    pub height: u32,
    /// The sample aspect ratio, `None` when unknown.
    pub sample_aspect: Option<(u32, u32)>,
    /// The frame rate as `(numerator, denominator)`: the average rate, else
    /// the base rate; `None` when neither is known.
    pub frame_rate: Option<(u32, u32)>,
    /// The display rotation in degrees counterclockwise, when declared.
    pub rotation: Option<f64>,
    /// Frames a decoder outputs.
    pub frames: u64,
}

/// An audio stream, as its container declares it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioStream {
    /// The declared sample rate.
    pub rate: u32,
    /// The declared channel count.
    pub channels: u32,
}

/// Reads `encoded`'s container and streams through the demuxers `formats`
/// names, within `timeout`; a demuxer outside them, and any file or URL the
/// media names, is refused.
///
/// # Errors
///
/// [`Error::Codec`] when the media cannot be demuxed through `formats` or the
/// timeout passes.
pub fn probe_container(encoded: &[u8], formats: &[&str], timeout: Duration) -> Result<Container> {
    let timeout_ms = i64::try_from(timeout.as_millis()).unwrap_or(i64::MAX);
    let probed = ffi::probe_media(encoded, &formats.join(","), timeout_ms)?;
    let ratio = |(numerator, denominator): (i32, i32)| {
        let numerator = u32::try_from(numerator).ok().filter(|&value| value > 0)?;
        let denominator = u32::try_from(denominator).ok().filter(|&value| value > 0)?;
        Some((numerator, denominator))
    };
    Ok(Container {
        format_name: probed.format_name,
        video: probed.video.map(|video| VideoStream {
            codec: video.codec,
            width: video.width,
            height: video.height,
            sample_aspect: ratio(video.sample_aspect),
            frame_rate: ratio(video.average_rate).or_else(|| ratio(video.base_rate)),
            rotation: video.rotation,
            frames: video.frames,
        }),
        audio: probed
            .audio
            .map(|(rate, channels)| AudioStream { rate, channels }),
    })
}

/// An audio track as [`crate::read::read_audio`] decodes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodedTrack {
    /// The rate the track decodes at: its stream's declared rate.
    pub rate: u32,
    /// Its channels.
    pub channels: usize,
    /// Samples of each channel, encoder delay and padding trimmed.
    pub samples: usize,
}

/// Decodes the first audio track of `encoded` as the reader does and reports
/// its extent.
///
/// # Errors
///
/// [`Error::Plan`] when the media has no audio stream or one with more than
/// two channels; [`Error::Codec`] when decoding fails.
pub fn decoded_track(encoded: &[u8]) -> Result<DecodedTrack> {
    let decoded = ffi::decode_audio(encoded)?;
    Ok(DecodedTrack {
        rate: decoded.rate,
        channels: decoded.channels,
        samples: decoded.samples,
    })
}

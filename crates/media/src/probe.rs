//! Media probing: what a condition's media is, read by the decoders that read
//! it ([`crate::read`]), so the extent a request's transform states is the
//! extent the reader produces.
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
//! The media the readers read is declared here ([`readable`]): the image
//! formats and largest image, the audio and video demuxers and the video
//! codecs they are written and tested for. A caller refuses media outside them
//! before it reaches a reader, and what else a request accepts (rates,
//! durations, channel layouts) is the caller's rule. A video stream also
//! reports the size and aspect it is displayed at ([`VideoStream::displayed`]),
//! as FFmpeg displays it.

use std::time::Duration;

use crate::{Error, Result, ffi};

/// The media the condition readers ([`crate::read`]) read, by kind.
pub mod readable {
    /// The image formats [`crate::read::read_image`] decodes.
    pub const IMAGE_FORMATS: [image::ImageFormat; 3] = [
        image::ImageFormat::Jpeg,
        image::ImageFormat::Png,
        image::ImageFormat::WebP,
    ];

    /// The largest image [`crate::read::read_image`] decodes, in pixels: Pillow's
    /// decompression-bomb limit (twice its 89,478,485-pixel warning threshold), the limit of the
    /// reference image loader whose decoding the reader reproduces, so an image the reference
    /// refuses is refused here too.
    pub const IMAGE_PIXELS: u64 = 2 * 89_478_485;

    /// The demuxers audio conditions are read through, by FFmpeg's names: WAV and MP3.
    pub const AUDIO_DEMUXERS: [&str; 2] = ["wav", "mp3"];

    /// The demuxer reference videos are read through, by FFmpeg's name: ISO-BMFF/QuickTime, which
    /// reads MP4 and MOV.
    pub const VIDEO_DEMUXERS: [&str; 1] = ["mov"];

    /// The codecs of the video streams [`crate::read::read_video`] decodes, by FFmpeg's names:
    /// H.264 and H.265.
    pub const VIDEO_CODECS: [&str; 2] = ["h264", "hevc"];

    /// The most channels an audio track the reader decodes may have: it reads mono and stereo
    /// ([`crate::read::read_audio`] duplicates mono to stereo).
    pub const AUDIO_CHANNELS: u32 = 2;
}

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
    /// The coded pixel format, as FFmpeg names it (`yuv420p`); empty when
    /// unknown.
    pub pixel_format: String,
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

/// How a video stream is displayed: its coded size scaled by its sample aspect ratio and turned
/// by its display rotation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Displayed {
    /// The display aspect, width to height, in lowest terms.
    pub aspect: (u64, u64),
    /// The displayed width in pixels.
    pub width: u32,
    /// The displayed height in pixels.
    pub height: u32,
}

impl VideoStream {
    /// How the stream is displayed, as FFmpeg displays it: the coded size scaled by the sample
    /// aspect ratio (square when unknown), keeping the coded height and rounding the width half to
    /// even, then turned by the display rotation, a quarter turn swapping the sides.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a stream without a size or with a rotation that is not a multiple
    /// of 90 degrees.
    pub fn displayed(&self) -> Result<Displayed> {
        if self.width == 0 || self.height == 0 {
            return Err(Error::invalid("the video stream has no size"));
        }
        let (sample_width, sample_height) = self.sample_aspect.unwrap_or((1, 1));
        let (width, height) = (
            u64::from(self.width) * u64::from(sample_width),
            u64::from(self.height) * u64::from(sample_height),
        );
        let divisor = num_integer::gcd(width, height);
        let mut aspect = (width / divisor, height / divisor);
        // Coded sizes stay far below 2^53, so the arithmetic is exact up to the rounding.
        let scaled = f64::from(self.width) * f64::from(sample_width) / f64::from(sample_height);
        let mut size = (scaled.round_ties_even() as u32, self.height);

        if let Some(rotation) = self.rotation {
            let turns = rotation / 90.0;
            if (turns - turns.round()).abs() > 1e-6 {
                return Err(Error::Invalid(format!(
                    "a video rotation must be a multiple of 90 degrees, got {rotation}"
                )));
            }
            if turns.round().rem_euclid(2.0) == 1.0 {
                aspect = (aspect.1, aspect.0);
                size = (size.1, size.0);
            }
        }
        Ok(Displayed {
            aspect,
            width: size.0,
            height: size.1,
        })
    }
}

/// An audio stream, as its container declares it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioStream {
    /// The codec, as FFmpeg names it (`aac`, `mp3`).
    pub codec: String,
    /// The declared sample rate.
    pub rate: u32,
    /// The declared channel count.
    pub channels: u32,
    /// The average bit rate, as the container states it or the demuxer
    /// measures it from the stream's packets; `None` when unknown.
    pub bit_rate: Option<u64>,
    /// The stream's duration as the container states it; `None` when
    /// unknown.
    pub duration: Option<Duration>,
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
            pixel_format: video.pixel_format,
            width: video.width,
            height: video.height,
            sample_aspect: ratio(video.sample_aspect),
            frame_rate: ratio(video.average_rate).or_else(|| ratio(video.base_rate)),
            rotation: video.rotation,
            frames: video.frames,
        }),
        audio: probed.audio.map(|audio| AudioStream {
            codec: audio.codec,
            rate: audio.rate,
            channels: audio.channels,
            bit_rate: audio.bit_rate,
            duration: audio.duration_us.map(Duration::from_micros),
        }),
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
/// [`Error::Invalid`] when the media has no audio stream or one with more than
/// two channels; [`Error::Codec`] when decoding fails.
pub fn decoded_track(encoded: &[u8]) -> Result<DecodedTrack> {
    let decoded = ffi::decode_audio(encoded)?;
    Ok(DecodedTrack {
        rate: decoded.rate,
        channels: decoded.channels,
        samples: decoded.samples,
    })
}

#[cfg(test)]
mod tests {
    use super::{Displayed, VideoStream};

    fn h264() -> VideoStream {
        VideoStream {
            codec: "h264".to_owned(),
            pixel_format: "yuv420p".to_owned(),
            width: 320,
            height: 180,
            sample_aspect: Some((1, 1)),
            frame_rate: Some((30000, 1001)),
            rotation: None,
            frames: 75,
        }
    }

    fn displayed(stream: VideoStream) -> Displayed {
        stream.displayed().unwrap()
    }

    /// The displayed size and aspect follow the sample aspect ratio, square when unknown, and a
    /// quarter turn swaps them.
    #[test]
    fn a_stream_is_displayed_scaled_and_turned() {
        let plain = Displayed {
            aspect: (16, 9),
            width: 320,
            height: 180,
        };
        assert_eq!(displayed(h264()), plain);
        assert_eq!(
            displayed(VideoStream {
                sample_aspect: None,
                ..h264()
            }),
            plain
        );
        assert_eq!(
            displayed(VideoStream {
                rotation: Some(180.0),
                ..h264()
            }),
            plain
        );
        for rotation in [90.0, -90.0, 270.0] {
            assert_eq!(
                displayed(VideoStream {
                    rotation: Some(rotation),
                    ..h264()
                }),
                Displayed {
                    aspect: (9, 16),
                    width: 180,
                    height: 320,
                },
                "{rotation}"
            );
        }
        // 352x288 at a 12:11 sample aspect displays at 4:3, 384 pixels wide.
        assert_eq!(
            displayed(VideoStream {
                width: 352,
                height: 288,
                sample_aspect: Some((12, 11)),
                ..h264()
            }),
            Displayed {
                aspect: (4, 3),
                width: 384,
                height: 288,
            }
        );
    }

    /// A stream without a size, or turned by other than quarter turns, has no displayed size.
    #[test]
    fn a_stream_without_a_size_or_a_quarter_turn_is_refused() {
        let sizeless = VideoStream { width: 0, ..h264() };
        assert!(sizeless.displayed().is_err());
        let slanted = VideoStream {
            rotation: Some(45.0),
            ..h264()
        };
        assert!(slanted.displayed().is_err());
    }
}

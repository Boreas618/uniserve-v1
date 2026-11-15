//! Media probing: the facts of a condition's media that its processing needs.
//!
//! The facts are read by the decoders that read the media for the request
//! (`media::probe`), so the extent a request is processed with is the extent the
//! worker's media reader produces. This module states which media a request
//! accepts:
//!
//! - An image must be JPEG, PNG or WEBP of at most [`MAX_IMAGE_PIXELS`]
//!   pixels. Its size is the displayed one, its EXIF orientation applied
//!   (orientations 5 to 8 swap the sides).
//! - A video must be an MP4/MOV (ISO-BMFF/QuickTime) file with an H.264 or
//!   H.265 stream at 23.976 to 60 frames per second. Its display aspect is the
//!   coded size scaled by the sample aspect ratio and turned by the display
//!   matrix rotation, as FFmpeg displays it; its frame count is the number of
//!   frames a decoder outputs (packets an edit list discards excluded); its
//!   first audio stream is its soundtrack.
//! - An audio file must be WAV or MP3, mono or stereo; its sample count is
//!   the number of samples the media reader decodes, encoder delay and padding
//!   trimmed.
//!
//! Audio and video are read in memory, only through the allowed demuxers, and
//! a demuxer may open no other file or URL, so a crafted playlist or concat
//! script cannot make the probe read anything but the bytes it was given.
//! Every rejection names `input_references[i]`.

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use tokio::sync::Semaphore;

use processing::{
    AudioMetadata, FrameRate, ImageMetadata, InputError, MediaMetadata, ReferenceKind,
    VideoMetadata,
};

/// Refuses `input_references[index]` with `message`.
fn refuse(index: usize, message: impl Into<String>) -> InputError {
    InputError::invalid(format!("input_references[{index}]"), message)
}

/// A failure of the prober's own resources.
fn internal(message: impl Into<String>) -> InputError {
    InputError::Internal(message.into())
}

/// The largest image accepted, in pixels: Pillow, which decodes condition
/// images for the conditioner and the VAE, refuses larger ones as
/// decompression bombs.
pub const MAX_IMAGE_PIXELS: u64 = 2 * 89_478_485;

/// Video frame rates accepted, as `[23.976, 60]` frames per second.
const MIN_FRAME_RATE_MILLIS: u64 = 23_976;
const MAX_FRAME_RATE: u64 = 60;

/// Demuxers each condition type may be read with: ISO-BMFF/QuickTime for
/// video, WAV and MP3 for audio.
const VIDEO_FORMATS: &[&str] = &["mov"];
const AUDIO_FORMATS: &[&str] = &["wav", "mp3"];

/// The longest one probe of a file may take before the media is rejected.
const PROBE_TIMEOUT: Duration = Duration::from_secs(30);

/// Reads the displayed size of a JPEG, PNG or WEBP image.
///
/// # Errors
///
/// Returns a message for another format, an unreadable header, or more than
/// [`MAX_IMAGE_PIXELS`] pixels.
pub fn probe_image(bytes: &[u8]) -> Result<ImageMetadata, String> {
    let facts = media::probe::probe_image(bytes)
        .map_err(|_| "is not a well-formed JPEG, PNG or WEBP image".to_owned())?;
    if !matches!(
        facts.format,
        image::ImageFormat::Jpeg | image::ImageFormat::Png | image::ImageFormat::WebP
    ) {
        return Err("is not a JPEG, PNG or WEBP image".to_owned());
    }
    let (width, height) = (facts.width, facts.height);
    if width == 0 || height == 0 {
        return Err("the image has no size".to_owned());
    }
    if u64::from(width) * u64::from(height) > MAX_IMAGE_PIXELS {
        return Err(format!(
            "the image has {width}x{height} pixels, more than {MAX_IMAGE_PIXELS}"
        ));
    }
    Ok(ImageMetadata { width, height })
}

/// Probes condition media; shared by all requests.
pub struct MediaProber {
    permits: Arc<Semaphore>,
}

impl MediaProber {
    /// Creates a prober that probes at most `max_concurrent_probes`
    /// conditions at once across all requests.
    pub fn new(max_concurrent_probes: usize) -> Self {
        let permits = Arc::new(Semaphore::new(max_concurrent_probes.max(1)));
        Self { permits }
    }

    /// Creates a prober that probes one condition per available CPU at once.
    pub fn per_cpu() -> Self {
        Self::new(std::thread::available_parallelism().map_or(1, usize::from))
    }

    /// Probes the media of `input_references[index]`, of kind `kind`.
    ///
    /// # Errors
    ///
    /// Returns [`InputError::Invalid`] naming `input_references[index]` for
    /// media of the wrong type or outside the accepted formats, and
    /// [`InputError::Internal`] when the probe cannot run.
    pub async fn probe(
        &self,
        index: usize,
        kind: ReferenceKind,
        bytes: &Bytes,
    ) -> Result<MediaMetadata, InputError> {
        let _permit = self
            .permits
            .acquire()
            .await
            .map_err(|_| internal("the media prober is shut down"))?;
        let bytes = bytes.clone();
        // Probing reads and partly decodes the media: blocking work, run on
        // the blocking pool.
        tokio::task::spawn_blocking(move || probe_media(kind, &bytes))
            .await
            .map_err(|error| internal(format!("the media probe failed: {error}")))?
            .map_err(|message| refuse(index, message))
    }
}

/// Probes media of kind `kind` against the request rules.
fn probe_media(kind: ReferenceKind, bytes: &[u8]) -> Result<MediaMetadata, String> {
    let unreadable = |error: media::Error| format!("the media cannot be read: {error}");
    match kind {
        ReferenceKind::Image => probe_image(bytes).map(MediaMetadata::Image),
        ReferenceKind::Audio => {
            let container = media::probe::probe_container(bytes, AUDIO_FORMATS, PROBE_TIMEOUT)
                .map_err(unreadable)?;
            let stream = audio_stream(&container, true)?;
            let track = media::probe::decoded_track(bytes).map_err(unreadable)?;
            Ok(MediaMetadata::Audio(stream.facts(&track)?))
        }
        ReferenceKind::Video => {
            let container = media::probe::probe_container(bytes, VIDEO_FORMATS, PROBE_TIMEOUT)
                .map_err(unreadable)?;
            let (video, soundtrack) = video_streams(&container)?;
            if video.frames == 0 {
                return Err("the video stream has no frames".to_owned());
            }
            let soundtrack = match soundtrack {
                Some(stream) => {
                    let track = media::probe::decoded_track(bytes).map_err(unreadable)?;
                    Some(stream.facts(&track)?)
                }
                None => None,
            };
            Ok(MediaMetadata::Video(VideoMetadata {
                display_width: video.display.0,
                display_height: video.display.1,
                width: video.size.0,
                height: video.size.1,
                frame_rate: video.frame_rate,
                frames: video.frames,
                soundtrack,
            }))
        }
    }
}

const fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}

/// The primary video stream of a probed file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VideoFacts {
    /// The display aspect in lowest terms.
    display: (u64, u64),
    /// The displayed size in pixels.
    size: (u32, u32),
    frame_rate: FrameRate,
    frames: u64,
}

/// An audio stream of a probed file, before its samples are counted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AudioFacts {
    sample_rate: u32,
}

impl AudioFacts {
    /// The track's facts, from the reader's decode of it.
    fn facts(self, track: &media::probe::DecodedTrack) -> Result<AudioMetadata, String> {
        if track.rate != self.sample_rate {
            return Err(format!(
                "the audio stream declares {} Hz and decodes at {} Hz",
                self.sample_rate, track.rate
            ));
        }
        if track.samples == 0 {
            return Err("the audio stream has no samples".to_owned());
        }
        Ok(AudioMetadata {
            sample_rate: self.sample_rate,
            samples: track.samples as u64,
        })
    }
}

/// The first audio stream: required of an audio file, the soundtrack of a
/// video when present. It must be mono or stereo at a positive rate.
fn audio_stream(
    container: &media::probe::Container,
    audio_file: bool,
) -> Result<AudioFacts, String> {
    if audio_file && !(container.is_format("wav") || container.is_format("mp3")) {
        return Err(format!(
            "an audio condition must be WAV or MP3, got {:?}",
            container.format_name
        ));
    }
    let stream = container
        .audio
        .as_ref()
        .ok_or_else(|| "the media has no audio stream".to_owned())?;
    if stream.rate == 0 {
        return Err("the audio stream has no sample rate".to_owned());
    }
    match stream.channels {
        1 | 2 => Ok(AudioFacts {
            sample_rate: stream.rate,
        }),
        channels => Err(format!(
            "an audio track must be mono or stereo, got {channels} channels"
        )),
    }
}

/// The primary video stream and the soundtrack of an MP4/MOV file.
fn video_streams(
    container: &media::probe::Container,
) -> Result<(VideoFacts, Option<AudioFacts>), String> {
    if !(container.is_format("mov") || container.is_format("mp4")) {
        return Err(format!(
            "a video condition must be MP4 or MOV, got {:?}",
            container.format_name
        ));
    }
    let stream = container
        .video
        .as_ref()
        .ok_or_else(|| "the media has no video stream".to_owned())?;
    if !matches!(stream.codec.as_str(), "h264" | "hevc") {
        return Err(format!(
            "a video must be H.264 or H.265, got {:?}",
            stream.codec
        ));
    }
    let (width, height) = (stream.width, stream.height);
    if width == 0 || height == 0 {
        return Err("the video stream has no size".to_owned());
    }

    let (numerator, denominator) = stream
        .frame_rate
        .ok_or_else(|| "the video stream has no frame rate".to_owned())?;
    let (num, den) = (u64::from(numerator), u64::from(denominator));
    if num * 1000 < MIN_FRAME_RATE_MILLIS * den || num > MAX_FRAME_RATE * den {
        return Err(format!(
            "a video must run at 23.976 to 60 frames per second, got {:.3}",
            num as f64 / den as f64
        ));
    }

    // FFmpeg displays the coded size scaled by the sample aspect ratio and
    // turned by the display matrix; an unknown ratio is square.
    let (sample_width, sample_height) = stream.sample_aspect.unwrap_or((1, 1));
    let mut display = (
        u64::from(width) * u64::from(sample_width),
        u64::from(height) * u64::from(sample_height),
    );
    let divisor = gcd(display.0, display.1);
    display = (display.0 / divisor, display.1 / divisor);
    // The displayed size keeps the coded height and scales the width; coded
    // sizes stay far below 2^53, so the arithmetic is exact up to rounding.
    let scaled = f64::from(width) * f64::from(sample_width) / f64::from(sample_height);
    let mut size = (scaled.round_ties_even() as u32, height);
    if let Some(rotation) = stream.rotation {
        let turns = rotation / 90.0;
        if (turns - turns.round()).abs() > 1e-6 {
            return Err(format!(
                "a video rotation must be a multiple of 90 degrees, got {rotation}"
            ));
        }
        if turns.round().rem_euclid(2.0) == 1.0 {
            display = (display.1, display.0);
            size = (size.1, size.0);
        }
    }

    let soundtrack = match container.audio {
        Some(_) => Some(audio_stream(container, false)?),
        None => None,
    };
    Ok((
        VideoFacts {
            display,
            size,
            frame_rate: FrameRate {
                numerator,
                denominator,
            },
            frames: stream.frames,
        },
        soundtrack,
    ))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use bytes::Bytes;
    use image::{ImageEncoder as _, RgbImage};
    use media::probe::{AudioStream, Container, VideoStream};

    use super::{
        AudioMetadata, FrameRate, ImageMetadata, InputError, MediaMetadata, MediaProber,
        ReferenceKind, VideoMetadata, audio_stream, probe_image, video_streams,
    };

    /// A TIFF-structured EXIF block holding one orientation entry of the
    /// given type.
    fn exif(orientation: u32, little_endian: bool, kind: u16) -> Vec<u8> {
        let u16_bytes = |value: u16| {
            if little_endian {
                value.to_le_bytes()
            } else {
                value.to_be_bytes()
            }
        };
        let u32_bytes = |value: u32| {
            if little_endian {
                value.to_le_bytes()
            } else {
                value.to_be_bytes()
            }
        };
        let mut block = b"Exif\0\0".to_vec();
        block.extend(if little_endian { *b"II*\0" } else { *b"MM\0*" });
        block.extend(u32_bytes(8));
        block.extend(u16_bytes(2));
        // An unrelated entry precedes the orientation (ImageWidth).
        block.extend(u16_bytes(0x0100));
        block.extend(u16_bytes(4));
        block.extend(u32_bytes(1));
        block.extend(u32_bytes(640));
        block.extend(u16_bytes(0x0112));
        block.extend(u16_bytes(kind));
        block.extend(u32_bytes(1));
        match kind {
            3 => {
                block.extend(u16_bytes(orientation as u16));
                block.extend([0, 0]);
            }
            _ => block.extend(u32_bytes(orientation)),
        }
        block.extend(u32_bytes(0));
        block
    }

    fn pixels(width: u32, height: u32) -> RgbImage {
        RgbImage::from_fn(width, height, |x, y| image::Rgb([x as u8, y as u8, 7]))
    }

    fn jpeg(width: u32, height: u32, exif: Option<&[u8]>) -> Vec<u8> {
        let mut encoded = Vec::new();
        image::codecs::jpeg::JpegEncoder::new(&mut encoded)
            .write_image(
                &pixels(width, height),
                width,
                height,
                image::ExtendedColorType::Rgb8,
            )
            .unwrap();
        if let Some(exif) = exif {
            // APP1 right after SOI; the length counts itself.
            let mut segment = vec![0xFF, 0xE1];
            segment.extend(u16::try_from(exif.len() + 2).unwrap().to_be_bytes());
            segment.extend(exif);
            encoded.splice(2..2, segment);
        }
        encoded
    }

    fn crc32(bytes: &[u8]) -> u32 {
        let mut crc = !0_u32;
        for &byte in bytes {
            crc ^= u32::from(byte);
            for _ in 0..8 {
                crc = if crc & 1 == 1 {
                    (crc >> 1) ^ 0xEDB8_8320
                } else {
                    crc >> 1
                };
            }
        }
        !crc
    }

    fn png_chunk(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
        let mut chunk = u32::try_from(data.len()).unwrap().to_be_bytes().to_vec();
        chunk.extend(kind);
        chunk.extend(data);
        let mut crc_input = kind.to_vec();
        crc_input.extend(data);
        chunk.extend(crc32(&crc_input).to_be_bytes());
        chunk
    }

    /// A PNG with an `eXIf` chunk before the image data, or after it.
    fn png(width: u32, height: u32, exif: Option<&[u8]>, after_data: bool) -> Vec<u8> {
        let mut encoded = Vec::new();
        image::codecs::png::PngEncoder::new(&mut encoded)
            .write_image(
                &pixels(width, height),
                width,
                height,
                image::ExtendedColorType::Rgb8,
            )
            .unwrap();
        if let Some(exif) = exif {
            let tiff = exif.strip_prefix(b"Exif\0\0").unwrap();
            let chunk = png_chunk(b"eXIf", tiff);
            // IHDR ends at byte 33; IEND is the last 12 bytes.
            let at = if after_data { encoded.len() - 12 } else { 33 };
            encoded.splice(at..at, chunk);
        }
        encoded
    }

    fn riff(chunks: &[(&[u8; 4], Vec<u8>)]) -> Vec<u8> {
        let mut body = b"WEBP".to_vec();
        for (kind, data) in chunks {
            body.extend(*kind);
            body.extend(u32::try_from(data.len()).unwrap().to_le_bytes());
            body.extend(data);
            if data.len() % 2 == 1 {
                body.push(0);
            }
        }
        let mut file = b"RIFF".to_vec();
        file.extend(u32::try_from(body.len()).unwrap().to_le_bytes());
        file.extend(body);
        file
    }

    /// A lossless WEBP, optionally wrapped in an extended container with
    /// an EXIF chunk.
    fn webp(width: u32, height: u32, exif: Option<&[u8]>) -> Vec<u8> {
        let mut encoded = Vec::new();
        image::codecs::webp::WebPEncoder::new_lossless(&mut encoded)
            .write_image(
                &pixels(width, height),
                width,
                height,
                image::ExtendedColorType::Rgb8,
            )
            .unwrap();
        let Some(exif) = exif else {
            return encoded;
        };
        let bitstream = encoded[20..].to_vec();
        let mut extended = vec![0x08, 0, 0, 0];
        extended.extend(&(width - 1).to_le_bytes()[..3]);
        extended.extend(&(height - 1).to_le_bytes()[..3]);
        riff(&[
            (b"VP8X", extended),
            (b"VP8L", bitstream),
            (b"EXIF", exif.to_vec()),
        ])
    }

    /// The raster the media reader decodes `encoded` to: the `image`
    /// decoder's pixels with their orientation applied.
    fn decoded_size(encoded: &[u8]) -> (u32, u32) {
        use image::ImageDecoder as _;

        let mut decoder = image::ImageReader::new(std::io::Cursor::new(encoded))
            .with_guessed_format()
            .unwrap()
            .into_decoder()
            .unwrap();
        let orientation = decoder.orientation().unwrap();
        let mut decoded = image::DynamicImage::from_decoder(decoder).unwrap();
        decoded.apply_orientation(orientation);
        (decoded.width(), decoded.height())
    }

    /// The displayed size is the raster the reader decodes the image to, in
    /// every format, byte order and orientation tag type; orientations 5 to 8
    /// transpose a JPEG.
    #[test]
    fn images_report_the_size_they_decode_to() {
        for orientation in 0..=9 {
            for (little_endian, kind) in [(true, 3), (false, 3), (true, 4), (false, 4)] {
                let block = exif(orientation, little_endian, kind);
                for encoded in [
                    jpeg(64, 40, Some(&block)),
                    png(64, 40, Some(&block), false),
                    png(64, 40, Some(&block), true),
                    webp(64, 40, Some(&block)),
                ] {
                    let facts = probe_image(&encoded).unwrap();
                    assert_eq!(
                        (facts.width, facts.height),
                        decoded_size(&encoded),
                        "orientation {orientation}"
                    );
                }
            }
            let transposed = (5..=8).contains(&orientation);
            let expected = if transposed { (40, 64) } else { (64, 40) };
            let facts = probe_image(&jpeg(64, 40, Some(&exif(orientation, true, 3)))).unwrap();
            assert_eq!((facts.width, facts.height), expected, "{orientation}");
        }
        for encoded in [
            jpeg(33, 17, None),
            png(33, 17, None, false),
            webp(33, 17, None),
        ] {
            assert_eq!(
                probe_image(&encoded).unwrap(),
                ImageMetadata {
                    width: 33,
                    height: 17
                }
            );
        }
    }

    /// Other formats, malformed headers and decompression bombs are
    /// rejected.
    #[test]
    fn images_outside_the_rules_are_rejected() {
        let mut gif = Vec::new();
        image::codecs::gif::GifEncoder::new(&mut gif)
            .encode(&[0; 12], 2, 2, image::ExtendedColorType::Rgb8)
            .unwrap();
        assert!(probe_image(&gif).is_err());
        assert!(probe_image(b"not an image").is_err());
        assert!(probe_image(&jpeg(16, 16, None)[..20]).is_err());
        assert!(probe_image(&png(16, 16, None, false)[..20]).is_err());

        // A header may declare a size no decoder would allocate.
        let mut bomb = png(16, 16, None, false);
        bomb[16..20].copy_from_slice(&20_000_u32.to_be_bytes());
        bomb[20..24].copy_from_slice(&20_000_u32.to_be_bytes());
        let crc = crc32(&bomb[12..29]);
        bomb[29..33].copy_from_slice(&crc.to_be_bytes());
        assert!(probe_image(&bomb).is_err());
    }

    const MOV: &str = "mov,mp4,m4a,3gp,3g2,mj2";

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

    fn aac(channels: u32) -> AudioStream {
        AudioStream {
            codec: "aac".to_owned(),
            rate: 44100,
            channels,
            bit_rate: None,
            duration: None,
        }
    }

    fn container(
        format: &str,
        video: Option<VideoStream>,
        audio: Option<AudioStream>,
    ) -> Container {
        Container {
            format_name: format.to_owned(),
            video,
            audio,
        }
    }

    /// A video's display aspect and size follow its sample aspect ratio and
    /// display rotation; its first audio stream is its soundtrack.
    #[test]
    fn video_streams_report_display_aspect_and_soundtrack() {
        let (video, soundtrack) =
            video_streams(&container(MOV, Some(h264()), Some(aac(1)))).unwrap();
        assert_eq!(video.display, (16, 9));
        assert_eq!(video.size, (320, 180));
        assert_eq!(
            video.frame_rate,
            FrameRate {
                numerator: 30000,
                denominator: 1001
            }
        );
        assert_eq!(video.frames, 75);
        assert_eq!(soundtrack.unwrap().sample_rate, 44100);

        for rotation in [90.0, -90.0, 270.0] {
            let stream = VideoStream {
                rotation: Some(rotation),
                ..h264()
            };
            let (video, soundtrack) = video_streams(&container(MOV, Some(stream), None)).unwrap();
            assert_eq!(video.display, (9, 16));
            assert_eq!(video.size, (180, 320));
            assert!(soundtrack.is_none());
        }
        let upside_down = VideoStream {
            rotation: Some(180.0),
            ..h264()
        };
        let output = container(MOV, Some(upside_down), None);
        assert_eq!(video_streams(&output).unwrap().0.display, (16, 9));

        // HEVC 352x288 at a 12:11 sample aspect displays at 4:3, 384 pixels
        // wide.
        let hevc = VideoStream {
            codec: "hevc".to_owned(),
            width: 352,
            height: 288,
            sample_aspect: Some((12, 11)),
            frame_rate: Some((25, 1)),
            ..h264()
        };
        let (video, _) = video_streams(&container(MOV, Some(hevc), None)).unwrap();
        assert_eq!((video.display, video.size), ((4, 3), (384, 288)));

        // An unknown sample aspect ratio is square.
        let unknown = VideoStream {
            sample_aspect: None,
            ..h264()
        };
        let (video, _) = video_streams(&container(MOV, Some(unknown), None)).unwrap();
        assert_eq!(video.display, (16, 9));
    }

    /// Containers, codecs, frame rates, rotations and channel layouts
    /// outside the rules are rejected.
    #[test]
    fn video_streams_outside_the_rules_are_rejected() {
        let with = |change: fn(&mut VideoStream)| {
            let mut stream = h264();
            change(&mut stream);
            Some(stream)
        };
        let cases = [
            container("matroska,webm", Some(h264()), None),
            container(MOV, with(|stream| stream.codec = "mpeg4".to_owned()), None),
            container(MOV, with(|stream| stream.frame_rate = Some((15, 1))), None),
            container(MOV, with(|stream| stream.frame_rate = Some((120, 1))), None),
            container(
                MOV,
                with(|stream| stream.frame_rate = Some((2997, 126))),
                None,
            ),
            container(MOV, with(|stream| stream.frame_rate = None), None),
            container(MOV, with(|stream| stream.rotation = Some(45.0)), None),
            container(MOV, with(|stream| stream.width = 0), None),
            container(MOV, Some(h264()), Some(aac(6))),
            container(MOV, None, Some(aac(2))),
        ];
        for case in cases {
            assert!(video_streams(&case).is_err(), "{case:?}");
        }
        // 23.976 and 60 frames per second are both accepted.
        for rate in [(2997, 125), (24000, 1001), (60, 1)] {
            let case = container(MOV, with_rate(rate), None);
            assert!(video_streams(&case).is_ok(), "{rate:?}");
        }
    }

    fn with_rate(rate: (u32, u32)) -> Option<VideoStream> {
        Some(VideoStream {
            frame_rate: Some(rate),
            ..h264()
        })
    }

    /// Audio files must be WAV or MP3 with a mono or stereo stream.
    #[test]
    fn audio_streams_follow_the_rules() {
        let wav = container(
            "wav",
            None,
            Some(AudioStream {
                rate: 48000,
                ..aac(2)
            }),
        );
        assert_eq!(audio_stream(&wav, true).unwrap().sample_rate, 48000);
        let mp3 = container("mp3", None, Some(aac(1)));
        assert_eq!(audio_stream(&mp3, true).unwrap().sample_rate, 44100);

        assert!(audio_stream(&container("flac", None, Some(aac(2))), true).is_err());
        assert!(audio_stream(&container("wav", None, Some(aac(6))), true).is_err());
        assert!(audio_stream(&container("wav", None, None), true).is_err());
        let silent = AudioStream { rate: 0, ..aac(2) };
        assert!(audio_stream(&container("wav", None, Some(silent)), true).is_err());
    }

    /// Media of another kind than its reference names, or that no allowed
    /// demuxer reads, is refused naming its reference.
    #[tokio::test]
    async fn unreadable_media_names_its_reference() {
        let prober = MediaProber::new(1);
        for (kind, bytes) in [
            (ReferenceKind::Audio, Bytes::from_static(b"RIFF")),
            (ReferenceKind::Video, Bytes::from(png(16, 16, None, false))),
            (ReferenceKind::Image, Bytes::from_static(b"not an image")),
        ] {
            let error = prober.probe(2, kind, &bytes).await.unwrap_err();
            assert!(
                matches!(&error, InputError::Invalid { param, .. } if param == "input_references[2]"),
                "{error}"
            );
        }
    }

    /// The `ffprobe` named by `UNISERVE_FFPROBE` and the `ffmpeg` in its
    /// directory, which encodes the probed media; the `ffprobe` is the
    /// independent oracle of a decoder's frame count.
    ///
    /// # Panics
    ///
    /// When `UNISERVE_FFPROBE` is unset: the live test is ignored by default
    /// and fails when requested without its tools.
    fn ffmpeg_tools() -> (PathBuf, PathBuf) {
        let ffprobe = PathBuf::from(std::env::var_os("UNISERVE_FFPROBE").expect(
            "UNISERVE_FFPROBE must name an ffprobe beside an ffmpeg with libx264, libx265 and libmp3lame",
        ));
        let ffmpeg = ffprobe.with_file_name("ffmpeg");
        (ffprobe, ffmpeg)
    }

    fn encode(ffmpeg: &PathBuf, directory: &std::path::Path, name: &str, arguments: &str) -> Bytes {
        let output = directory.join(name);
        let status = std::process::Command::new(ffmpeg)
            .args(["-v", "error", "-y"])
            .args(arguments.split_whitespace())
            .arg(&output)
            .status()
            .unwrap();
        assert!(status.success(), "ffmpeg {arguments}");
        Bytes::from(std::fs::read(output).unwrap())
    }

    /// Probing real media reports what a decoder outputs.
    #[tokio::test]
    #[ignore = "requires UNISERVE_FFPROBE: an ffprobe beside an ffmpeg with libx264, libx265 and libmp3lame"]
    async fn probing_reports_decoded_media() {
        let (ffprobe, ffmpeg) = ffmpeg_tools();
        let directory = tempfile::tempdir().unwrap();
        let prober = MediaProber::new(4);
        let path = directory.path();
        let video = "-f lavfi -i testsrc2=size=320x180:rate=30000/1001 \
                     -f lavfi -i sine=frequency=440:sample_rate=44100:duration=2.5 \
                     -frames:v 75 -c:v libx264 -pix_fmt yuv420p -c:a pcm_s16le";
        let plain = encode(&ffmpeg, path, "plain.mov", video);
        let rotated = encode(
            &ffmpeg,
            path,
            "rotated.mov",
            &format!(
                "-display_rotation 90 -i {} -c copy",
                path.join("plain.mov").display()
            ),
        );
        let trimmed = encode(
            &ffmpeg,
            path,
            "trimmed.mp4",
            &format!(
                "-ss 0.5 -i {} -c:v copy -an",
                path.join("plain.mov").display()
            ),
        );
        let wav = encode(
            &ffmpeg,
            path,
            "tone.wav",
            "-f lavfi -i sine=frequency=440:sample_rate=48000:duration=3.3 -ac 2",
        );

        let soundtrack = AudioMetadata {
            sample_rate: 44100,
            samples: 110_250,
        };
        let expected = MediaMetadata::Video(VideoMetadata {
            display_width: 16,
            display_height: 9,
            width: 320,
            height: 180,
            frame_rate: FrameRate {
                numerator: 30000,
                denominator: 1001,
            },
            frames: 75,
            soundtrack: Some(soundtrack),
        });
        let probed = prober.probe(0, ReferenceKind::Video, &plain).await.unwrap();
        assert_eq!(probed, expected);
        let MediaMetadata::Video(probed) = prober
            .probe(0, ReferenceKind::Video, &rotated)
            .await
            .unwrap()
        else {
            panic!("not a video");
        };
        assert_eq!((probed.display_width, probed.display_height), (9, 16));
        assert_eq!((probed.width, probed.height), (180, 320));

        // An edit list that starts mid-GOP discards the leading frames; the
        // decoder's own count is the oracle.
        let MediaMetadata::Video(probed) = prober
            .probe(0, ReferenceKind::Video, &trimmed)
            .await
            .unwrap()
        else {
            panic!("not a video");
        };
        let counted = std::process::Command::new(&ffprobe)
            .args(["-v", "error", "-count_frames", "-select_streams", "v:0"])
            .args(["-show_entries", "stream=nb_read_frames", "-of", "csv=p=0"])
            .arg(path.join("trimmed.mp4"))
            .output()
            .unwrap();
        let decoded: u64 = String::from_utf8_lossy(&counted.stdout)
            .trim()
            .parse()
            .unwrap();
        assert_eq!(probed.frames, decoded);
        assert!(probed.frames < 75);
        assert!(probed.soundtrack.is_none());

        let expected = MediaMetadata::Audio(AudioMetadata {
            sample_rate: 48000,
            samples: 158_400,
        });
        assert_eq!(
            prober.probe(1, ReferenceKind::Audio, &wav).await.unwrap(),
            expected
        );

        // Media of the wrong kind names its condition.
        let error = prober
            .probe(3, ReferenceKind::Audio, &plain)
            .await
            .unwrap_err();
        assert!(
            matches!(&error, InputError::Invalid { param, .. } if param == "input_references[3]"),
            "{error}"
        );
        let error = prober
            .probe(4, ReferenceKind::Video, &wav)
            .await
            .unwrap_err();
        assert!(
            matches!(&error, InputError::Invalid { param, .. } if param == "input_references[4]"),
            "{error}"
        );

        // HEVC at a 12:11 sample aspect, and MP3 with its encoder delay and
        // padding trimmed, as a decoder outputs.
        let hevc = encode(
            &ffmpeg,
            path,
            "hevc.mp4",
            "-f lavfi -i testsrc2=size=352x288:rate=25 -frames:v 50 -vf setsar=12/11 \
             -c:v libx265 -tag:v hvc1",
        );
        let MediaMetadata::Video(probed) =
            prober.probe(0, ReferenceKind::Video, &hevc).await.unwrap()
        else {
            panic!("not a video");
        };
        assert_eq!(
            (probed.display_width, probed.display_height, probed.frames),
            (4, 3, 50)
        );
        let mp3 = encode(
            &ffmpeg,
            path,
            "tone.mp3",
            "-f lavfi -i sine=frequency=440:sample_rate=44100:duration=3.3 -c:a libmp3lame",
        );
        let expected = MediaMetadata::Audio(AudioMetadata {
            sample_rate: 44100,
            samples: 145_530,
        });
        assert_eq!(
            prober.probe(1, ReferenceKind::Audio, &mp3).await.unwrap(),
            expected
        );
    }
}

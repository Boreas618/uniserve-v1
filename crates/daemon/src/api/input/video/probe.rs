//! Media probing: the facts of a condition's media that its processing needs.
//!
//! The facts are read by the decoders that read the media for the request
//! (`media::probe`), so the extent a request is processed with is the extent the
//! worker's media reader produces, and a medium is accepted when the reader
//! reads it (`media::probe::readable`):
//!
//! - An image is one of the reader's formats (JPEG, PNG or WEBP) of at most the
//!   pixels it decodes. Its size is the displayed one, its EXIF orientation
//!   applied (orientations 5 to 8 swap the sides).
//! - A video is a file of the reader's demuxer (ISO-BMFF/QuickTime: MP4 or MOV)
//!   with a stream of one of its codecs (H.264 or H.265). Its display aspect and
//!   size are the stream's as FFmpeg displays it (`VideoStream::displayed`); its
//!   frame count is the number of frames a decoder outputs (packets an edit list
//!   discards excluded); its first audio stream is its soundtrack. The frame
//!   rates a model takes are its request rules'.
//! - An audio file is one of the reader's demuxers (WAV or MP3) with at most the
//!   channels it decodes (mono or stereo); its sample count is the number of
//!   samples the media reader decodes, encoder delay and padding trimmed.
//!
//! Audio and video are read in memory, only through the reader's demuxers, and
//! a demuxer may open no other file or URL, so a crafted playlist or concat
//! script cannot make the probe read anything but the bytes it was given. A
//! demuxer's read is bounded by the media table's `probe_timeout`. Every
//! rejection names `input_references[i]`.

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use media::probe::readable;
use proto::config::{MediaDefaults, MediaSources};
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

/// Reads the displayed size of an image the reader decodes.
///
/// # Errors
///
/// Returns a message for another format, an unreadable header, or more pixels than the reader
/// decodes.
pub fn probe_image(bytes: &[u8]) -> Result<ImageMetadata, String> {
    let facts = media::probe::probe_image(bytes)
        .map_err(|_| "is not a well-formed JPEG, PNG or WEBP image".to_owned())?;
    if !readable::IMAGE_FORMATS.contains(&facts.format) {
        return Err("is not a JPEG, PNG or WEBP image".to_owned());
    }
    let (width, height) = (facts.width, facts.height);
    if width == 0 || height == 0 {
        return Err("the image has no size".to_owned());
    }
    if u64::from(width) * u64::from(height) > readable::IMAGE_PIXELS {
        return Err(format!(
            "the image has {width}x{height} pixels, more than {}",
            readable::IMAGE_PIXELS
        ));
    }
    Ok(ImageMetadata { width, height })
}

/// Probes condition media; shared by all requests.
pub struct MediaProber {
    permits: Arc<Semaphore>,
    /// The longest one demuxer read of a medium may take before the medium is refused.
    timeout: Duration,
}

impl MediaProber {
    /// The prober of a media table: at most its `max_concurrent_probes` media at once across
    /// all requests, the daemon's available parallelism (its CPU set and quota) when unset,
    /// each demuxer read bounded by its `probe_timeout`.
    ///
    /// # Errors
    ///
    /// Fails when the table states no concurrency and the available parallelism cannot be read.
    pub fn new<D: MediaDefaults>(options: &MediaSources<D>) -> anyhow::Result<Self> {
        let concurrent = match options.max_concurrent_probes {
            Some(concurrent) => concurrent,
            None => std::thread::available_parallelism()
                .map_err(|error| anyhow::anyhow!("reading the available parallelism: {error}"))?,
        };
        Ok(Self {
            permits: Arc::new(Semaphore::new(concurrent.get())),
            timeout: options.probe_timeout,
        })
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
        let timeout = self.timeout;
        // Probing reads and partly decodes the media: blocking work, run on
        // the blocking pool.
        tokio::task::spawn_blocking(move || probe_media(kind, &bytes, timeout))
            .await
            .map_err(|error| internal(format!("the media probe failed: {error}")))?
            .map_err(|message| refuse(index, message))
    }
}

/// Probes media of kind `kind` against what the reader reads, each demuxer read within
/// `timeout`.
fn probe_media(
    kind: ReferenceKind,
    bytes: &[u8],
    timeout: Duration,
) -> Result<MediaMetadata, String> {
    let unreadable = |error: media::Error| format!("the media cannot be read: {error}");
    match kind {
        ReferenceKind::Image => probe_image(bytes).map(MediaMetadata::Image),
        ReferenceKind::Audio => {
            let container =
                media::probe::probe_container(bytes, &readable::AUDIO_DEMUXERS, timeout)
                    .map_err(unreadable)?;
            let stream = audio_stream(&container)?;
            let track = media::probe::decoded_track(bytes).map_err(unreadable)?;
            Ok(MediaMetadata::Audio(stream.facts(&track)?))
        }
        ReferenceKind::Video => {
            let container =
                media::probe::probe_container(bytes, &readable::VIDEO_DEMUXERS, timeout)
                    .map_err(unreadable)?;
            let video = video_stream(&container)?;
            let soundtrack = match container.audio {
                Some(_) => {
                    let stream = audio_stream(&container)?;
                    let track = media::probe::decoded_track(bytes).map_err(unreadable)?;
                    Some(stream.facts(&track)?)
                }
                None => None,
            };
            Ok(MediaMetadata::Video(VideoMetadata {
                soundtrack,
                ..video
            }))
        }
    }
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

/// The first audio stream of a file the reader's demuxer read: an audio file's track, or a
/// video's soundtrack. It has a positive rate and at most the channels the reader decodes.
fn audio_stream(container: &media::probe::Container) -> Result<AudioFacts, String> {
    let stream = container
        .audio
        .as_ref()
        .ok_or_else(|| "the media has no audio stream".to_owned())?;
    if stream.rate == 0 {
        return Err("the audio stream has no sample rate".to_owned());
    }
    if stream.channels == 0 || stream.channels > readable::AUDIO_CHANNELS {
        return Err(format!(
            "an audio track must be mono or stereo, got {} channels",
            stream.channels
        ));
    }
    Ok(AudioFacts {
        sample_rate: stream.rate,
    })
}

/// The primary video stream of a file the reader's demuxer read, without its soundtrack: a
/// stream of one of the reader's codecs, with a frame rate and frames, as it is displayed.
fn video_stream(container: &media::probe::Container) -> Result<VideoMetadata, String> {
    let stream = container
        .video
        .as_ref()
        .ok_or_else(|| "the media has no video stream".to_owned())?;
    if !readable::VIDEO_CODECS.contains(&stream.codec.as_str()) {
        return Err(format!(
            "a video must be H.264 or H.265, got {:?}",
            stream.codec
        ));
    }
    let (numerator, denominator) = stream
        .frame_rate
        .ok_or_else(|| "the video stream has no frame rate".to_owned())?;
    if stream.frames == 0 {
        return Err("the video stream has no frames".to_owned());
    }
    let displayed = stream.displayed().map_err(|error| error.to_string())?;
    Ok(VideoMetadata {
        display_width: displayed.aspect.0,
        display_height: displayed.aspect.1,
        width: displayed.width,
        height: displayed.height,
        frame_rate: FrameRate {
            numerator,
            denominator,
        },
        frames: stream.frames,
        soundtrack: None,
    })
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use bytes::Bytes;
    use image::{ImageEncoder as _, RgbImage};
    use media::probe::{AudioStream, Container, VideoStream};
    use proto::config::MediaConfig;

    use super::{
        AudioMetadata, FrameRate, ImageMetadata, InputError, MediaMetadata, MediaProber,
        ReferenceKind, VideoMetadata, audio_stream, probe_image, video_stream,
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

    /// A PNG of `width` x `height` pixels: its header states the size and its image data holds
    /// the pixels, with an `eXIf` chunk before the image data, or after it.
    fn png(width: u32, height: u32, exif: Option<&[u8]>, after_data: bool) -> Vec<u8> {
        let mut encoded = Vec::new();
        let mut encoder = png::Encoder::new(&mut encoded, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        let tiff = exif.map(|exif| exif.strip_prefix(b"Exif\0\0").unwrap());
        if let Some(tiff) = tiff.filter(|_| !after_data) {
            writer.write_chunk(png::chunk::eXIf, tiff).unwrap();
        }
        writer.write_image_data(&pixels(width, height)).unwrap();
        if let Some(tiff) = tiff.filter(|_| after_data) {
            writer.write_chunk(png::chunk::eXIf, tiff).unwrap();
        }
        writer.finish().unwrap();
        encoded
    }

    /// A PNG whose header states `width` x `height` pixels and whose image data is empty: a header
    /// that states a size no decoder would allocate.
    fn png_header(width: u32, height: u32) -> Vec<u8> {
        let mut encoded = Vec::new();
        let mut encoder = png::Encoder::new(&mut encoded, width, height);
        encoder.set_color(png::ColorType::Rgb);
        let mut writer = encoder.write_header().unwrap();
        writer.write_chunk(png::chunk::IDAT, &[]).unwrap();
        // Dropping the writer ends the file.
        drop(writer);
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
        let error = probe_image(&png_header(20_000, 20_000)).unwrap_err();
        assert!(error.contains("pixels"), "{error}");
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
            rotation: Some(90.0),
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

    fn container(video: Option<VideoStream>, audio: Option<AudioStream>) -> Container {
        Container {
            format_name: MOV.to_owned(),
            video,
            audio,
        }
    }

    /// A video reads as its stream is displayed, with its frame rate and frames.
    #[test]
    fn a_video_stream_reads_as_it_is_displayed() {
        assert_eq!(
            video_stream(&container(Some(h264()), None)).unwrap(),
            VideoMetadata {
                display_width: 9,
                display_height: 16,
                width: 180,
                height: 320,
                frame_rate: FrameRate {
                    numerator: 30000,
                    denominator: 1001
                },
                frames: 75,
                soundtrack: None,
            }
        );
    }

    /// A video of a codec the reader does not decode, without a frame rate, frames or a displayed
    /// size, or a file without a video stream, is refused.
    #[test]
    fn video_streams_the_reader_does_not_read_are_refused() {
        let with = |change: fn(&mut VideoStream)| {
            let mut stream = h264();
            change(&mut stream);
            Some(stream)
        };
        let cases = [
            container(with(|stream| stream.codec = "mpeg4".to_owned()), None),
            container(with(|stream| stream.frame_rate = None), None),
            container(with(|stream| stream.frames = 0), None),
            container(with(|stream| stream.rotation = Some(45.0)), None),
            container(with(|stream| stream.width = 0), None),
            container(None, Some(aac(2))),
        ];
        for case in cases {
            assert!(video_stream(&case).is_err(), "{case:?}");
        }
    }

    /// An audio track is mono or stereo at a positive rate.
    #[test]
    fn audio_streams_are_mono_or_stereo_at_a_rate() {
        let stereo = AudioStream {
            rate: 48000,
            ..aac(2)
        };
        assert_eq!(
            audio_stream(&container(None, Some(stereo))).unwrap(),
            super::AudioFacts { sample_rate: 48000 }
        );
        assert_eq!(
            audio_stream(&container(None, Some(aac(1))))
                .unwrap()
                .sample_rate,
            44100
        );

        assert!(audio_stream(&container(None, Some(aac(6)))).is_err());
        assert!(audio_stream(&container(None, None)).is_err());
        let silent = AudioStream { rate: 0, ..aac(2) };
        assert!(audio_stream(&container(None, Some(silent))).is_err());
    }

    /// A prober of one probe at a time.
    fn prober() -> MediaProber {
        MediaProber::new(&MediaConfig {
            max_concurrent_probes: Some(NonZeroUsize::MIN),
            ..MediaConfig::default()
        })
        .unwrap()
    }

    /// Media of another kind than its reference names, or that no reader's demuxer reads, is
    /// refused naming its reference.
    #[tokio::test]
    async fn unreadable_media_names_its_reference() {
        let prober = prober();
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

    /// An MP4 of `frames` frames of 320x180 H.264 at 24 frames per second, encoded and muxed with
    /// a stereo AAC soundtrack at 32 kHz by the media crate's own encoders.
    fn reference_video(frames: usize) -> Bytes {
        let (width, height, frame_rate, audio_rate) = (320_u32, 180_u32, 24_u32, 32_000_u32);
        let pixels: Vec<u8> = (0..frames * (width * height * 3) as usize)
            .map(|byte| (byte % 251) as u8)
            .collect();
        let unit = media::encode_video_unit(
            media::Rgb24Frames {
                pixels: &pixels,
                height,
                width,
            },
            frame_rate,
        )
        .unwrap();
        let samples = media::audio_samples(frames, frame_rate, audio_rate);
        let tone: Vec<i16> = (0..samples * 2)
            .map(|sample| (sample % 64) as i16)
            .collect();
        let track = media::encode_audio_track(&tone, samples, audio_rate).unwrap();
        let mut container = media::Container::open(&unit, frame_rate, audio_rate).unwrap();
        container.append(&unit, 0).unwrap();
        Bytes::from(container.finish(&track).unwrap())
    }

    /// Probing a real MP4 reports what the reader decodes: the frames it was encoded with, its
    /// rate and displayed size, and its soundtrack's decoded samples; the same file is not audio
    /// a reader takes.
    #[tokio::test]
    async fn probing_reports_decoded_media() {
        let video = reference_video(48);
        let MediaMetadata::Video(probed) = prober()
            .probe(0, ReferenceKind::Video, &video)
            .await
            .unwrap()
        else {
            panic!("not a video");
        };
        let decoded = media::probe::decoded_track(&video).unwrap();
        assert_eq!(
            probed,
            VideoMetadata {
                display_width: 16,
                display_height: 9,
                width: 320,
                height: 180,
                frame_rate: FrameRate {
                    numerator: 24,
                    denominator: 1
                },
                frames: 48,
                soundtrack: Some(AudioMetadata {
                    sample_rate: 32_000,
                    samples: decoded.samples as u64,
                }),
            }
        );

        let error = prober()
            .probe(3, ReferenceKind::Audio, &video)
            .await
            .unwrap_err();
        assert!(
            matches!(&error, InputError::Invalid { param, .. } if param == "input_references[3]"),
            "{error}"
        );
    }
}

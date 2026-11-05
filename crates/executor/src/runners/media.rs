//! The media group's host work: a clip encoded into its unit, a track encoded into AAC, and the
//! request's MP4 assembled from them.
//!
//! `crates/media` holds the codecs, the muxer and one assembly per request (`MediaRunner`'s
//! `MuxSession`); this module is what the executor owes it - the two published products' bytes, and
//! the request's two published products' bytes. **The clip this reads is the postprocessor's output,
//! `[frames, rows, columns, 3]` uint8** - the `video_decoder` component's second entry point, which
//! drops the VAE's padding frames, cross-fades the clip seams and de-normalises the RGB - so nothing
//! here converts a raster: an encoder's input and a postprocessor's output are the same bytes.
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
    /// Samples per second of the track: `32000` for MiniMax-H3's BigVGAN.
    pub audio_rate: u32,
}

impl Rates {
    /// The samples one request's soundtrack covers, from the unit list the request was opened with.
    ///
    /// **The unit list is the timeline's own** - `VideoPostprocessor::splices_of` is its one home -
    /// so this reads it rather than recomputing a schedule: a units function here would be a second
    /// statement of the checkpoint's `17 * n + 5`, and the muxer is assembled from the first one.
    ///
    /// The artifact's timeline is frames at the frame rate and samples at the audio rate, so the
    /// two counts are one number read twice: a track shorter than this leaves the last frames
    /// silent, and a longer one is samples the container has no place for.
    pub fn samples(&self, units: &[u32]) -> usize {
        let frames: u64 = units.iter().map(|&frames| u64::from(frames)).sum();
        (frames * u64::from(self.audio_rate) / u64::from(self.frame_rate)) as usize
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
    /// `units` is the request's timeline as the model's schedule states it - the kept frame count
    /// of each clip, in order - which fixes the assembly the first call opens. `clip` is the bytes
    /// the `video_decoder` component published -
    /// `[frames, height, width, 3]` uint8, the postprocessor's own order - and `shape` is
    /// `(frames, height, width)` of this unit. Nothing is converted here: a postprocessor that
    /// de-normalised the raster and an encoder that takes RGB24 agree on the bytes.
    pub fn encode_clip(
        &self,
        request: RequestKey,
        units: &[u32],
        shape: (u32, u32, u32),
        clip: &[u8],
    ) -> Result<()> {
        let (unit_frames, height, width) = shape;
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
                units,
            )?;
        }
        let decoded = unit_frames as usize * height as usize * width as usize * 3;
        if clip.len() != decoded {
            return Err(crate::Error::Invalid(format!(
                "a unit of {unit_frames} frame(s) of {width}x{height} RGB24 is {decoded} byte(s) \
                 and the clip carries {}: the raster the postprocessor spliced and the raster the \
                 encoder was told of disagree",
                clip.len()
            )));
        }
        let row = Rates::encoded_unit_bytes(unit_frames, height, width);
        let mut framed = vec![0_u8; row];
        let written = self.media.encode_unit(
            media::Rgb24Frames {
                pixels: clip,
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
    fn encoded_unit_bytes(frames: u32, height: u32, width: u32) -> usize {
        media::encoded_unit_bytes(frames, height, width)
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

    /// The soundtrack covers every frame of every unit at the artifact's rate.
    ///
    /// The two counts are one number read twice - the timeline is frames at the frame rate and
    /// samples at the audio rate - so a track shorter than this leaves the last frames silent and a
    /// longer one is samples the container has no place for.
    #[test]
    fn the_track_covers_every_frame_of_every_unit() {
        let rates = Rates {
            frame_rate: 24,
            audio_rate: 32_000,
        };
        assert_eq!(rates.samples(&[17, 17, 22]), 56 * 32_000 / 24);
        assert_eq!(rates.samples(&[]), 0);
    }

    /// A clip that is not `frames * height * width * 3` bytes is refused before an encoder sees it.
    ///
    /// The postprocessor writes RGB24 by definition, so a clip of another size is a raster that
    /// disagrees with the timeline the assembly was opened for - and an encoder handed it would
    /// encode whatever the shorter buffer held.
    #[test]
    fn a_clip_that_disagrees_with_the_unit_raster_is_refused() {
        let output = Output::new(
            std::sync::Arc::new(media::MediaRunner::new(media::MediaConfig::default())),
            Rates {
                frame_rate: 24,
                audio_rate: 32_000,
            },
        );
        let request = RequestKey::new(0, ipc::RequestId(1), 0);
        let error = output
            .encode_clip(request, &[17, 22], (17, 2, 2), &[0_u8; 16])
            .expect_err("a short clip");
        let message = format!("{error}");
        assert!(
            message.contains("204") && message.contains("16"),
            "the refusal names both byte counts: {message}"
        );
    }
}

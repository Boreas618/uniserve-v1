//! The media group's host work: a clip encoded into its unit, a track encoded into AAC, and the
//! request's MP4 assembled from them.
//!
//! `crates/media` holds the codecs, the muxer and one assembly per request (`MediaRunner`'s
//! `MuxSession`); this module is what the executor owes it - the request's two published products'
//! bytes, and the assembly they go into. **The clip this reads is the postprocessor's output,
//! `[frames, rows, columns, 3]` uint8** - the `video_decoder` component's second entry point, which
//! drops the VAE's padding frames, cross-fades the clip seams and de-normalises the RGB - so nothing
//! here converts a raster: an encoder's input and a postprocessor's output are the same bytes.
//!
//! Everything here is host work on blocking threads by design (`specs/worker.md` §4.5): an
//! encoding, a mux and a reader are not device computations, and the executor's `Media` lane is
//! where they run. [`MediaWork`] is that lane's action: one stage of one request's artifact, with
//! the bytes the stage reads already in host memory. It has no graph, no bucket and no device
//! stream, which is why the three media calls a `t2va` request ends with are not device variants.

use std::sync::Arc;

use ipc::RequestKey;

use crate::Result;
use crate::host::{HostAction, Outcome};

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

/// One stage of one request's artifact, as the process's `Media` lane runs it.
///
/// §2.4's fourth kind of blocking work is "encoding, muxing and reading media", and this is the
/// action that kind was declared for: it owns the bytes its stage reads - already in host memory,
/// ordered after whatever produced them - and answers the assembly's bytes when its stage ends the
/// request. A lane keeps one action per admitted stage, so a slow encode never holds a device
/// stream, a scheduler thread or a module load.
///
/// **The three stages are the request's three media calls after its decode:** a unit's frames into
/// an encoded unit, the track's samples into AAC, and the assembly's end into the MP4's bytes. The
/// unit list is passed with every stage because it is what the assembly was opened with, and the
/// model's schedule is its one home.
pub struct MediaWork {
    /// The request whose assembly this stage appends to.
    request: RequestKey,
    /// The request's units, as the model's schedule states them.
    units: Vec<u32>,
    /// Which stage this is, and the extent it works at.
    stage: Stage,
    /// The published bytes this stage reads, and nothing for the stage that reads none: the mux's
    /// inputs are the two encodes the deployment's output already holds.
    input: Option<platform::PinnedBuffer>,
    /// The media group's host encoder, one per deployment.
    output: Arc<Output>,
}

/// The three stages of a request's artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// One unit's `[frames, rows, columns, 3]` frames into the encoded unit it covers.
    Unit {
        /// Frames this unit covers.
        frames: u32,
        /// Raster rows of a frame.
        height: u32,
        /// Raster columns of a frame.
        width: u32,
    },
    /// The track's `[samples, 2]` integers into AAC.
    ///
    /// The count is the buffer's - `bytes / (2 * size_of::<i16>())` - rather than a number the
    /// stage carries: the audio decoder states a window's length (`pcm_window`, the checkpoint's own
    /// multiplication), so a second computation here would be a different number waiting to
    /// disagree by one at the end of a request.
    Track,
    /// The end of the request's assembly: the MP4's bytes are the stage's answer.
    Mux,
}

impl MediaWork {
    /// A stage of `request`'s artifact over `input`.
    ///
    /// `input` is the product the stage reads - the unit's frames for a unit, the track for a
    /// track - and `None` for the mux, whose inputs are the encodes this action's `output` already
    /// holds. A mux stage handed bytes would be a caller stating an input the muxer never reads.
    pub fn new(
        output: Arc<Output>,
        request: RequestKey,
        units: Vec<u32>,
        stage: Stage,
        input: Option<platform::PinnedBuffer>,
    ) -> Self {
        Self {
            request,
            units,
            stage,
            input,
            output,
        }
    }

    /// The bytes this stage reads, or the refusal that names a stage with none.
    fn bytes(&self) -> Result<&[u8]> {
        let input = self.input.as_ref().ok_or_else(|| {
            crate::Error::Invalid(format!(
                "{:?} reads the product it was built from and this action carries no bytes",
                self.stage
            ))
        })?;
        // SAFETY: the bytes were copied into this buffer after their producer's write was waited
        // out, and the action owns the buffer: nothing else writes it while `run` reads it.
        Ok(unsafe { input.as_slice() })
    }
}

impl HostAction for MediaWork {
    /// The MP4's bytes when this stage ends the request, and nothing otherwise.
    type Output = Option<Vec<u8>>;

    fn run(&self) -> Result<Self::Output> {
        match self.stage {
            Stage::Unit {
                frames,
                height,
                width,
            } => {
                let bytes = self.bytes()?;
                self.output.encode_clip(
                    self.request,
                    &self.units,
                    (frames, height, width),
                    bytes,
                )?;
                Ok(None)
            }
            Stage::Track => {
                let bytes = self.bytes()?;
                let samples = bytes.len() / (2 * size_of::<i16>());
                self.output.encode_track(self.request, bytes, samples)?;
                Ok(None)
            }
            Stage::Mux => Ok(Some(self.output.finish(self.request)?)),
        }
    }

    /// Nothing to release: the bytes are this action's own host buffer, and dropping the action
    /// returns them. `stage` resolves them to the encoder's copy inside `run`.
    fn release(&self) -> Result<()> {
        Ok(())
    }

    /// The producer stopped successfully by construction: a stage is built from bytes that were
    /// copied here after their producer's write was waited out (`Products::stage`), so there is no
    /// outstanding device work that could still address them.
    fn input_outcome(&self) -> Outcome {
        Outcome::Success(())
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
    /// The deployment's one encoder, over the FFmpeg installation this process can reach.
    ///
    /// A deployment states the rates and nothing else: the installation is the environment's
    /// (`MediaConfig::from_env`, which is `UNISERVE_FFMPEG` or the prefix the crate was built
    /// against), so a plan constructs this in one call and the lane actions borrow it.
    ///
    /// **The codecs are required here rather than at the first request.** A host whose FFmpeg was
    /// built without the two encoders the artifact needs cannot serve a video at all, and the refusal
    /// belongs where the deployment is assembled - named, before any weight is loaded - instead of at
    /// the end of a request whose decode, denoise and audio all succeeded.
    pub fn from_env(rates: Rates) -> Result<Self> {
        let media = media::MediaRunner::new(media::MediaConfig::from_env());
        media.require_codecs()?;
        Ok(Self::new(Arc::new(media), rates))
    }

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
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| i16::from_ne_bytes(*pair))
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

    /// A request's two units and its track assemble into one MP4.
    ///
    /// This is the artifact a served `t2va` request ends with, and the one thing this module can
    /// prove without a device: the three stages `MediaWork` runs, in the order the engine runs them,
    /// over the bytes a postprocessor and an audio decode publish. The assertions are the
    /// container's own - its file type box, its media-data box, and a length no single unit's bytes
    /// reach - rather than this code's idea of what an encoder produced.
    ///
    /// It needs the runtime's `libx264` and `aac`, as `crates/media`'s own tests do: a host without
    /// them cannot serve a video at all, and `Output::from_env` refuses there by name.
    #[test]
    fn a_requests_units_and_track_assemble_into_one_mp4() {
        let output = Output::from_env(Rates {
            frame_rate: 24,
            audio_rate: 32_000,
        })
        .expect("the runtime provides libx264 and aac");
        let request = RequestKey::new(0, ipc::RequestId(7), 0);
        let units = [17_u32, 22];
        let (height, width) = (16_u32, 16_u32);
        let mut unit_bytes = 0_usize;
        for frames in units {
            let pixels = vec![0x40_u8; frames as usize * height as usize * width as usize * 3];
            unit_bytes += pixels.len();
            output
                .encode_clip(request, &units, (frames, height, width), &pixels)
                .expect("a unit encodes");
        }
        // The track the session's timeline covers, by the container's own rule: the audio decoder
        // states its window with its own arithmetic (`pcm_window`), so this test states the bytes
        // and the stage counts them rather than dividing a frame count a second way.
        let samples = media::codec::audio_samples(
            units.iter().map(|&frames| frames as usize).sum::<usize>(),
            24,
            32_000,
        );
        let pcm = vec![0_u8; samples * 2 * size_of::<i16>()];
        output
            .encode_track(request, &pcm, samples)
            .expect("a track encodes");
        let mp4 = output.finish(request).expect("the assembly ends");
        assert_eq!(&mp4[4..8], b"ftyp", "an MP4 begins with its file type box");
        assert!(
            mp4.windows(4).any(|box_type| box_type == b"moov"),
            "the container holds its movie box"
        );
        assert!(
            mp4.windows(4).any(|box_type| box_type == b"mdat"),
            "the container holds its media data"
        );
        assert!(
            mp4.len() < unit_bytes,
            "a compressed artifact is smaller than the frames it was made from"
        );
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

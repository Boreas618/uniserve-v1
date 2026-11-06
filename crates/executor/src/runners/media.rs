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
//!
//! **A conditioned request starts here too.** Its media reading decodes the bytes its conditions
//! publish in shared memory ([`read_conditions`]) into three pinned regions, laid out as the model's
//! encoders read them ([`nn::capability::MediaFormat`]), and publishes them as products: the text
//! encoding reads the patch rows and the denoiser's first step the pixels and the samples. A pinned
//! buffer is an address the device reads directly, so nothing is copied onto the device to be read
//! once.

use std::os::unix::fs::FileExt as _;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ipc::RequestKey;
use nn::capability::MediaFormat;

use crate::Result;
use crate::batch::{Batch, BatchOutput, CallId, CallOutput, Job, Jobs, RowState};
use crate::host::{HostAction, Lane, Outcome};
use crate::plan::Variant;
use crate::runner::{Binding, Runner, RunnerId};
use crate::runners::products::{Product, Products};

/// One request's conditions, as its media reading leaves them for the calls that read them.
///
/// Each region holds every condition of its kind one after another, in request order, and a reader
/// finds a condition's rows from the extents its conditioning states. The buffers are the storage
/// behind the views the store publishes (the store owns no memory), so they live until the request's
/// artifact is assembled.
#[derive(Default)]
pub struct ConditionRegions {
    /// Every visual condition's pixels, as `nn::capability::MediaInput::pixels` reads them: FP32.
    pixels: Option<platform::PinnedBuffer>,
    /// Every audio track, as `nn::capability::MediaInput::audio` reads it: FP32.
    samples: Option<platform::PinnedBuffer>,
    /// Every vision block's patch rows, in the model's patch type.
    patches: Option<platform::PinnedBuffer>,
}

/// The extents one request's regions take, from its conditioning alone.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RegionExtents {
    /// FP32 elements of the pixel region: one per byte of RGB the conditions decode to.
    pixels: u64,
    /// FP32 elements of the sample region.
    samples: u64,
    /// Patch rows of the vision region.
    patches: u64,
}

impl RegionExtents {
    /// What `conditioning`'s conditions take, as `format` lays them out.
    fn of(conditioning: &models::video::Conditioning, format: &dyn MediaFormat) -> Result<Self> {
        let mut extents = Self::default();
        for condition in &conditioning.conditions {
            extents.pixels += condition.pixel_bytes();
            if let Some(track) = condition.media.audio() {
                extents.samples += 2 * format.track_samples(track.samples)?;
            }
            if let Some(vision) = &condition.vision {
                extents.patches += vision.grid.patches();
            }
        }
        Ok(extents)
    }
}

/// What one host call reads a request's conditions with.
struct ConditionRead {
    /// The request's plan of them.
    conditioning: models::video::Conditioning,
    /// The regions the read fills, shared with the stage that publishes them when it ends.
    regions: Arc<Mutex<ConditionRegions>>,
    /// How the model's encoders read what the media decodes to.
    format: Arc<dyn MediaFormat>,
}

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
    /// What a media reading reads, and nothing for any other stage.
    read: Option<ConditionRead>,
}

/// The three stages of a request's artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// One unit's `[frames, rows, columns, 3]` frames into the encoded unit it covers.
    Unit {
        /// The unit's place in the request's timeline.
        index: u32,
        /// Frames this unit covers.
        frames: u32,
        /// Raster rows of a frame.
        height: u32,
        /// Raster columns of a frame.
        width: u32,
    },
    /// A trajectory's opening, which has nothing to encode: the call is answered as it arrives.
    Open,
    /// The track's `[samples, 2]` integers into AAC.
    ///
    /// The count is the buffer's - `bytes / (2 * size_of::<i16>())` - rather than a number the
    /// stage carries: the audio decoder states a window's length (`pcm_window`, the checkpoint's own
    /// multiplication), so a second computation here would be a different number waiting to
    /// disagree by one at the end of a request.
    Track,
    /// A muxing call that hands over encoded rounds. Each unit joined the assembly when its encode
    /// ran, so nothing is left to do and the call is answered as it arrives.
    Hand,
    /// The end of the request's assembly: the MP4's bytes are the stage's answer.
    Mux,
    /// A request's condition media decoded into its regions.
    Read,
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
            read: None,
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
        let outcome = match self.stage {
            Stage::Unit {
                index,
                frames,
                height,
                width,
            } => {
                let bytes = self.bytes()?;
                self.output.encode_clip(
                    self.request,
                    &self.units,
                    index as usize,
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
            Stage::Open | Stage::Hand => Ok(None),
            Stage::Read => {
                let read = self.read.as_ref().ok_or_else(|| {
                    crate::Error::Invalid(
                        "a media reading was staged with no conditions to read".into(),
                    )
                })?;
                let mut regions = crate::lock(&read.regions);
                read_conditions(
                    &self.output.media,
                    self.output.rates,
                    &read.conditioning,
                    read.format.as_ref(),
                    &mut regions,
                )?;
                Ok(None)
            }
        };
        if outcome.is_err() {
            // A stage that fails ends its request short of an artifact, so the assembly this
            // request opened is closed here: it lives in `crates/media`'s runner, one mux session
            // per request, and nothing else reaches it - the engine aborts the request and its
            // calls are never served again, so a session left open would hold a writer until the
            // stage is dropped. `discard` is a no-op for a request whose assembly never opened.
            self.output.discard(self.request);
        }
        outcome
    }

    /// Nothing to release: the bytes are this action's own host buffer, and dropping the action
    /// returns them. `stage` resolves them to the encoder's copy inside `run`. An assembly the
    /// action opened and did not finish is closed by `run`'s error path.
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
    /// containers, so the first clip of a request opens and every later one appends. The lock is
    /// held across the open, because the ranks encode a request's units at once and a unit must
    /// not reach an assembly another rank is still opening.
    opened: std::sync::Mutex<std::collections::HashSet<u64>>,
    /// Each request's encoded units on their way into its assembly.
    assemblies: std::sync::Mutex<std::collections::HashMap<u64, Assembly>>,
}

/// One request's encoded units on their way into its assembly.
///
/// The muxer takes units in the timeline's order and the ranks finish them in any order, so a unit
/// that arrives before its turn waits here, and the unit that completes a run takes the run in.
#[derive(Default)]
struct Assembly {
    /// The units the assembly holds, which is the next one's place.
    appended: usize,
    /// The framed units that arrived before their turn, by place.
    waiting: std::collections::BTreeMap<usize, Vec<u8>>,
    /// The encoded track, when it arrived before the first unit opened the assembly: the two encode
    /// beside each other, and only a unit knows the raster the assembly is opened at.
    track: Option<Vec<u8>>,
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
            assemblies: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// The rates every artifact of this output is built at.
    pub fn rates(&self) -> Rates {
        self.rates
    }

    /// Encodes one clip of `request` into the unit it covers.
    ///
    /// `units` is the request's timeline as the model's schedule states it - the kept frame count
    /// of each clip, in order - which fixes the assembly the first call opens. `index` is this
    /// unit's place in it: the ranks encode their own units at once, so the place is the batch's
    /// statement rather than a count of what arrived first. `clip` is the bytes
    /// the `video_decoder` component published -
    /// `[frames, height, width, 3]` uint8, the postprocessor's own order - and `shape` is
    /// `(frames, height, width)` of this unit. Nothing is converted here: a postprocessor that
    /// de-normalised the raster and an encoder that takes RGB24 agree on the bytes.
    pub fn encode_clip(
        &self,
        request: RequestKey,
        units: &[u32],
        index: usize,
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
        {
            let mut opened = self
                .opened
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if opened.insert(request_id(request)) {
                self.media.open_mux(
                    request_id(request),
                    width,
                    height,
                    self.rates.frame_rate,
                    self.rates.audio_rate,
                    units,
                )?;
                let track = crate::lock(&self.assemblies)
                    .get_mut(&request_id(request))
                    .and_then(|assembly| assembly.track.take());
                if let Some(track) = track {
                    self.media.set_audio(request_id(request), track)?;
                }
            }
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
        // The row is the store's framing, a length before the unit, and the assembly reads the unit.
        framed.truncate(written);
        framed.drain(..size_of::<u64>());
        let mut assemblies = crate::lock(&self.assemblies);
        let assembly = assemblies.entry(request_id(request)).or_default();
        assembly.waiting.insert(index, framed);
        while let Some(unit) = assembly.waiting.remove(&assembly.appended) {
            self.media.append_units(request_id(request), &[unit])?;
            assembly.appended += 1;
        }
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
        // The lock orders this against a unit opening the assembly: a track that finds none waits for
        // the unit, and the unit that opens it finds the track.
        let opened = self
            .opened
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if opened.contains(&request_id(request)) {
            self.media.set_audio(request_id(request), track)?;
        } else {
            crate::lock(&self.assemblies)
                .entry(request_id(request))
                .or_default()
                .track = Some(track);
        }
        Ok(())
    }

    /// Finishes `request`'s assembly and answers its MP4 bytes.
    pub fn finish(&self, request: RequestKey) -> Result<Vec<u8>> {
        self.opened
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(&request_id(request));
        crate::lock(&self.assemblies).remove(&request_id(request));
        Ok(self.media.finish_mux(request_id(request))?)
    }

    /// Drops `request`'s assembly, for a request that ends before it could be finished: a stage
    /// that fails calls this from `MediaWork::run`, so a failed call leaves no open mux session
    /// behind. Idempotent: `close_mux` removes a session if one exists.
    pub fn discard(&self, request: RequestKey) {
        self.opened
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(&request_id(request));
        crate::lock(&self.assemblies).remove(&request_id(request));
        self.media.close_mux(request_id(request));
    }
}

impl Rates {
    /// Bytes of one encoded unit's framed row at this raster.
    fn encoded_unit_bytes(frames: u32, height: u32, width: u32) -> usize {
        media::encoded_unit_bytes(frames, height, width)
    }
}

/// The bytes one condition's media was published as, read from shared memory.
fn published_bytes(source: &models::video::MediaLocator) -> Result<Vec<u8>> {
    let read = |error: std::io::Error| {
        crate::Error::Invalid(format!(
            "reading the published media {:?} of {} byte(s): {error}",
            source.name, source.bytes
        ))
    };
    let file = ipc::SharedMemory::open(&source.name, false).map_err(read)?;
    let length = usize::try_from(source.bytes)
        .map_err(|_| crate::Error::Invalid("a condition is larger than memory".into()))?;
    let mut bytes = vec![0_u8; length];
    file.read_exact_at(&mut bytes, 0).map_err(read)?;
    Ok(bytes)
}

/// `[frames, height, width, 3]` interleaved RGB as `[frames, 3, height, width]` planes.
fn planes_of(rgb: &[u8], frames: usize, height: usize, width: usize) -> Vec<u8> {
    let area = height * width;
    let mut planes = vec![0_u8; frames * 3 * area];
    for frame in 0..frames {
        let pixels = &rgb[frame * area * 3..(frame + 1) * area * 3];
        let plane = &mut planes[frame * 3 * area..(frame + 1) * 3 * area];
        for (index, pixel) in pixels.chunks_exact(3).enumerate() {
            plane[index] = pixel[0];
            plane[area + index] = pixel[1];
            plane[2 * area + index] = pixel[2];
        }
    }
    planes
}

/// One vision block's frames, `[frames, height, width, 3]` RGB at `raster`, as its patch rows.
///
/// The tower reads a grid's own pixels, so each frame is resized to the grid first: the reference's
/// `pack_pixels` does the same, in the same order (`uniserve/models/qwen3_vl/encoder.py`).
fn pack_block(
    format: &dyn MediaFormat,
    rgb: &[u8],
    frames: u32,
    raster: (u32, u32),
    grid: [u32; 3],
    out: &mut [u8],
) -> Result<()> {
    let (height, width) = raster;
    let side = format.patch_side();
    let (rows, columns) = (grid[1] * side, grid[2] * side);
    let planes = planes_of(rgb, frames as usize, height as usize, width as usize);
    let planes = if (rows, columns) == (height, width) {
        planes
    } else {
        let frame = 3 * height as usize * width as usize;
        let mut resized = Vec::with_capacity(frames as usize * 3 * (rows * columns) as usize);
        for source in planes.chunks_exact(frame) {
            resized.extend(media::resample::bicubic_antialias_u8(
                source, 3, height, width, rows, columns,
            )?);
        }
        resized
    };
    format
        .patches(&planes, frames, grid, out)
        .map_err(crate::Error::from)
}

/// Decodes one request's conditions into `regions`, in request order.
///
/// `crates/media` decodes the bytes into RGB frames and stereo PCM at their planned extents; the
/// model's format then lays them out as its encoders read them. A reference video's frames past its
/// encoded prefix are the vision tower's alone, which is why the reader hands them back.
fn read_conditions(
    media: &media::MediaRunner,
    rates: Rates,
    conditioning: &models::video::Conditioning,
    format: &dyn MediaFormat,
    regions: &mut ConditionRegions,
) -> Result<()> {
    use media::read::{AudioPlan, ConditionProducts, ConditionSource, ImagePlan, VideoClip};
    use models::video::ConditionMedia;

    let conditions = &conditioning.conditions;
    let bytes = conditions
        .iter()
        .map(|condition| published_bytes(&condition.source))
        .collect::<Result<Vec<_>>>()?;
    // A reference video is read by an external process, which takes a file: the shared memory
    // object the frontend published is one.
    let files = conditions
        .iter()
        .map(|condition| match condition.media {
            ConditionMedia::Video { .. } => ipc::SharedMemory::open(&condition.source.name, false)
                .map(Some)
                .map_err(|error| {
                    crate::Error::Invalid(format!(
                        "opening the published media {:?}: {error}",
                        condition.source.name
                    ))
                }),
            _ => Ok(None),
        })
        .collect::<Result<Vec<_>>>()?;

    let audio_plan = |track: &models::video::AudioClip| -> Result<AudioPlan> {
        let index = |value: u64| {
            usize::try_from(value)
                .map_err(|_| crate::Error::Invalid("a track is larger than memory".into()))
        };
        Ok(AudioPlan {
            native_rate: track.sample_rate,
            start: index(track.start_sample)?,
            source_samples: index(track.source_samples)?,
            samples: index(track.samples)?,
            rate: rates.audio_rate,
        })
    };
    let mut reads = Vec::with_capacity(conditions.len());
    let (mut pixel_rows, mut sample_rows) = (0_usize, 0_usize);
    for (index, condition) in conditions.iter().enumerate() {
        let (image, video) = match &condition.media {
            ConditionMedia::Image(fit) => (
                Some(ImagePlan {
                    resized: (fit.resized.width, fit.resized.height),
                    window: (fit.left, fit.top, fit.size.width, fit.size.height),
                }),
                None,
            ),
            ConditionMedia::Video { clip, .. } => (
                None,
                Some(VideoClip {
                    frame_rate: rates.frame_rate,
                    canvas: (clip.canvas.width, clip.canvas.height),
                    frames: clip.frames,
                    vae_frames: clip.vae_frames,
                    start_frame: clip.start_frame,
                }),
            ),
            ConditionMedia::Audio(_) => (None, None),
        };
        let pixels = usize::try_from(condition.pixel_bytes() / 3)
            .map_err(|_| crate::Error::Invalid("a condition is larger than memory".into()))?;
        let audio = condition.media.audio().map(audio_plan).transpose()?;
        let samples = audio.map_or(0, |plan| plan.samples);
        let source = match files[index].as_ref() {
            Some(file) => ConditionSource::File(file),
            None => ConditionSource::Bytes(&bytes[index]),
        };
        reads.push(media::read::ConditionRead {
            source,
            image,
            video,
            audio,
            pixels: pixel_rows..pixel_rows + pixels,
            samples: sample_rows..sample_rows + samples,
        });
        pixel_rows += pixels;
        sample_rows += samples;
    }

    let mut rgb = vec![0_u8; pixel_rows * 3];
    let mut pcm = vec![0_f32; sample_rows * 2];
    let decoded = media.read_conditions(
        &reads,
        &mut ConditionProducts {
            pixels: (pixel_rows > 0).then_some(rgb.as_mut_slice()),
            samples: (sample_rows > 0).then_some(pcm.as_mut_slice()),
        },
    )?;

    // Each condition's own rows, in the regions the encoders read.
    let mut pixels_at = 0_usize;
    let mut samples_at = 0_usize;
    let mut patches_at = 0_usize;
    let patch_bytes = usize::try_from(
        u64::from(format.patch_width())
            * format
                .patch_dtype()
                .bytes()
                .ok_or_else(|| crate::Error::Invalid("a patch row has no whole-byte size".into()))?,
    )
    .map_err(|_| crate::Error::Invalid("a patch row is larger than memory".into()))?;
    for (index, condition) in conditions.iter().enumerate() {
        let read = &reads[index];
        if let Some((frames, canvas)) = condition.media.pixels() {
            let source = &rgb[read.pixels.start * 3..read.pixels.end * 3];
            let region = regions
                .pixels
                .as_mut()
                .ok_or_else(|| crate::Error::Invalid("a condition's pixels have no region".into()))?;
            // SAFETY: nothing reads the region before this call publishes it, and it was allocated
            // for this request alone.
            let floats = unsafe { region.as_mut_slice() };
            let (_, floats, _) = unsafe { floats.align_to_mut::<f32>() };
            condition_rows(format, source, frames, canvas, &mut floats[pixels_at..])?;
            pixels_at += source.len();
        }
        if let Some(track) = condition.media.audio() {
            let padded = 2 * usize::try_from(format.track_samples(track.samples)?)
                .map_err(|_| crate::Error::Invalid("a track is larger than memory".into()))?;
            let region = regions
                .samples
                .as_mut()
                .ok_or_else(|| crate::Error::Invalid("a track has no region".into()))?;
            // SAFETY: as for the pixels above.
            let floats = unsafe { region.as_mut_slice() };
            let (_, floats, _) = unsafe { floats.align_to_mut::<f32>() };
            format.track(
                &pcm[read.samples.start * 2..read.samples.end * 2],
                track.samples,
                &mut floats[samples_at..samples_at + padded],
            )?;
            samples_at += padded;
        }
        if let Some(vision) = &condition.vision {
            let grid = [vision.grid.t, vision.grid.h, vision.grid.w];
            let rows = usize::try_from(vision.grid.patches())
                .map_err(|_| crate::Error::Invalid("a vision block is larger than memory".into()))?;
            let region = regions
                .patches
                .as_mut()
                .ok_or_else(|| crate::Error::Invalid("a vision block has no region".into()))?;
            // SAFETY: as for the pixels above.
            let out = unsafe { region.as_mut_slice() };
            let out = &mut out[patches_at * patch_bytes..(patches_at + rows) * patch_bytes];
            match decoded[index].as_ref() {
                // A video's block reads the frames its conditioner sampled from every kept frame,
                // and those frames are what the reader handed back.
                Some(video) => {
                    let (width, height) = video.clip.canvas;
                    let frame = width as usize * height as usize * 3;
                    let mut sampled = Vec::with_capacity(vision.frame_indices.len() * frame);
                    for index in &vision.frame_indices {
                        let start = *index as usize * frame;
                        sampled.extend_from_slice(&video.pixels[start..start + frame]);
                    }
                    let frames = u32::try_from(vision.frame_indices.len()).map_err(|_| {
                        crate::Error::Invalid("a video samples more frames than a u32".into())
                    })?;
                    pack_block(format, &sampled, frames, (height, width), grid, out)?;
                }
                None => {
                    let (frames, canvas) = condition.media.pixels().ok_or_else(|| {
                        crate::Error::Invalid(
                            "the conditioner reads a condition with no pixels".into(),
                        )
                    })?;
                    pack_block(
                        format,
                        &rgb[read.pixels.start * 3..read.pixels.end * 3],
                        frames,
                        (canvas.height, canvas.width),
                        grid,
                        out,
                    )?;
                }
            }
            patches_at += rows;
        }
    }
    Ok(())
}

/// Lays one visual condition's RGB out as its encoder reads it, at the head of `out`.
fn condition_rows(
    format: &dyn MediaFormat,
    rgb: &[u8],
    frames: u32,
    canvas: models::video::Canvas,
    out: &mut [f32],
) -> Result<()> {
    format
        .condition_pixels(rgb, frames, canvas.height, canvas.width, &mut out[..rgb.len()])
        .map_err(crate::Error::from)
}

/// A request's identity for the media crate, whose assembly is keyed by a number.
///
/// One deployment serves one epoch's requests, and the key already carries the epoch: the media
/// crate needs only the number within it.
fn request_id(request: RequestKey) -> u64 {
    request.request_id.0
}

/// The runner that serves a media request's three host calls.
///
/// §2.4 makes encoding, muxing and reading media the executor's fourth kind of blocking work, and
/// this is the runner those calls belong to: it owns no captured graph, no arena and no device
/// column, because none of the three calls computes on a device. What it owns is the request's two
/// published products' bytes and the assembly they go into.
///
/// **The two halves of a host call are as far apart as the executor's own protocol puts them.**
/// `submit` runs on the device thread when the batch reaches it, and it stages the product the call
/// reads into pinned host memory there - one device copy, ordered on this runner's stream after the
/// producer's write, which is the only place both ordering facts are known. `serve` runs when the
/// device executor reaches the job that `submit` returned, and it performs the stage on the
/// process's `Media` lane and waits for it: the bytes never touch a device stream, and a slow encode
/// holds a lane thread rather than a launch.
///
/// The artifact leaves as the call's `media_output`: the mux's bytes are published into a shared
/// memory object whose name crosses the wire, and the receiver unlinks it (`MediaSource::into_locator`),
/// so nothing here holds it after the call is answered.
pub struct MediaStage {
    /// The executor's name for this runner.
    id: RunnerId,
    /// The phases this runner serves, in scheduler order.
    phases: Vec<ipc::PhaseId>,
    /// The device the products it stages live on.
    device: platform::DeviceId,
    /// The stream its staging copies are ordered on.
    stream: platform::Stream,
    /// The media group's host encoder, one per deployment.
    output: Arc<Output>,
    /// The process's media lane: one action per admitted stage (2.4).
    lane: Lane<MediaWork>,
    /// The store the two products are read from, attached by the executor once.
    products: Option<Arc<dyn Products>>,
    /// The frames of each video unit of a request's timeline: the model's schedule, handed in as a
    /// function because it is the model's arithmetic and the executor holds no copy of it.
    units_of: fn(u32) -> Result<Vec<u32>>,
    /// What `submit` staged, one entry per call of the batch it was handed, and what `serve`
    /// performs call by call.
    ///
    /// A batch holds a call for each of its requests - the mechanism batches media requests as they
    /// arrive, and a host bucket states no bound (`Bucket::holds_computation`) - so staging one entry
    /// would leave every request after the first without an answer.
    pending: Vec<Prepared>,
    /// How the model's encoders read a request's decoded media.
    format: Arc<dyn MediaFormat>,
    /// The regions of the requests whose conditions were read, until their artifact is assembled:
    /// the store's views name this storage and own none of it.
    conditions: std::collections::HashMap<RequestKey, Arc<Mutex<ConditionRegions>>>,
}

/// One staged host call: the product's bytes, and the stage that reads them.
struct Prepared {
    /// The call this stage answers.
    call: CallId,
    /// The request whose assembly the stage appends to.
    request: RequestKey,
    /// The request's units, as the model's schedule states them.
    units: Vec<u32>,
    /// The stage and the extent it works at.
    stage: Stage,
    /// The product's bytes, and nothing for the mux, which reads what `output` holds.
    input: Option<platform::PinnedBuffer>,
    /// What a media reading reads, and nothing for any other stage.
    read: Option<ConditionRead>,
}

impl MediaStage {
    /// The runner of one deployment's three media calls.
    ///
    /// `units_of` is the model's own schedule - the frames of each video unit of a `frames`-frame
    /// timeline - passed rather than restated: a video unit's length is the checkpoint's arithmetic
    /// (`CLIP_FRAMES` and the seam), and an executor that computed it would be a second home for a
    /// number the model already states.
    pub fn new(
        binding: Binding,
        output: Arc<Output>,
        lane: Lane<MediaWork>,
        units_of: fn(u32) -> Result<Vec<u32>>,
        format: Arc<dyn MediaFormat>,
    ) -> Self {
        Self {
            id: binding.id,
            phases: binding.phases,
            device: binding.device,
            stream: binding.stream,
            output,
            lane,
            products: None,
            units_of,
            pending: Vec::new(),
            format,
            conditions: std::collections::HashMap::new(),
        }
    }

    /// The regions `conditioning`'s conditions are read into, and the read that fills them.
    fn prepare_read(&self, staged: &crate::batch::StagedRequest) -> Result<ConditionRead> {
        let conditioning =
            models::video::Conditioning::decode(&staged.conditioning).map_err(|error| {
                crate::Error::Invalid(format!(
                    "a media request's conditioning does not decode: {error}"
                ))
            })?;
        let extents = RegionExtents::of(&conditioning, self.format.as_ref())?;
        let region = |elements: u64, width: u64| -> Result<Option<platform::PinnedBuffer>> {
            if elements == 0 {
                return Ok(None);
            }
            let bytes = usize::try_from(elements * width)
                .map_err(|_| crate::Error::Invalid("a condition is larger than memory".into()))?;
            Ok(Some(platform::PinnedBuffer::new(self.device, bytes)?))
        };
        let row = u64::from(self.format.patch_width())
            * self
                .format
                .patch_dtype()
                .bytes()
                .ok_or_else(|| crate::Error::Invalid("a patch row has no whole-byte size".into()))?;
        let float = ops::DType::F32
            .bytes()
            .ok_or_else(|| crate::Error::Invalid("an FP32 has no whole-byte size".into()))?;
        Ok(ConditionRead {
            conditioning,
            regions: Arc::new(Mutex::new(ConditionRegions {
                pixels: region(extents.pixels, float)?,
                samples: region(extents.samples, float)?,
                patches: region(extents.patches, row)?,
            })),
            format: Arc::clone(&self.format),
        })
    }

    /// Publishes what a media reading left in `regions`, for the calls of `request` that read it.
    ///
    /// The host finished writing before this runs, and the writes are not on any stream, so the
    /// event is only what a crossing reader's ordering asks for: the store orders a reader on
    /// another stream after it.
    fn publish_regions(
        &mut self,
        request: RequestKey,
        regions: Arc<Mutex<ConditionRegions>>,
    ) -> Result<()> {
        let format = Arc::clone(&self.format);
        let products = Arc::clone(self.products()?);
        let stream_id = crate::arena::StreamId(self.stream.raw());
        {
            let held = crate::lock(&regions);
            let float = |buffer: &platform::PinnedBuffer| {
                ops::TensorView::new(
                    ops::DevicePtr::new(buffer.address()),
                    ops::DType::F32,
                    ops::Shape::new(&[buffer.len() as u64 / 4]),
                )
            };
            let patch = |buffer: &platform::PinnedBuffer| -> Result<ops::TensorView> {
                let dtype = format.patch_dtype();
                let width = u64::from(format.patch_width());
                let row = width * dtype.bytes().ok_or_else(|| {
                    crate::Error::Invalid("a patch row has no whole-byte size".into())
                })?;
                Ok(ops::TensorView::new(
                    ops::DevicePtr::new(buffer.address()),
                    dtype,
                    ops::Shape::new(&[buffer.len() as u64 / row, width]),
                ))
            };
            let views = [
                (Product::ConditionPixels, held.pixels.as_ref().map(float)),
                (Product::ConditionSamples, held.samples.as_ref().map(float)),
                (
                    Product::VisionPixels,
                    held.patches.as_ref().map(patch).transpose()?,
                ),
            ];
            for (kind, view) in views {
                let Some(view) = view else {
                    continue;
                };
                let written = crate::runners::products::written_on(&self.stream)?;
                products.publish(request, kind, self.device, stream_id, Some(written), &view)?;
            }
        }
        self.conditions.insert(request, regions);
        Ok(())
    }

    /// The store this runner reads the request's products from, and the refusal that names a
    /// deployment whose plan bound no such store.
    fn products(&self) -> Result<&Arc<dyn Products>> {
        self.products.as_ref().ok_or_else(|| {
            crate::Error::Invalid(
                "the media stage reads the request's published products and the executor attached \
                 no product store to it"
                    .into(),
            )
        })
    }

    /// Copies `kind` into pinned host memory, ordered after its producer's write.
    ///
    /// The copy happens here, on the device thread, because that is where both facts live: the
    /// product's byte length (`Products::product`) and the stream the copy is ordered on. A stage
    /// that deferred it would have to hold the request's region open across a lane thread's life.
    fn stage_product(&self, request: RequestKey, kind: Product) -> Result<platform::PinnedBuffer> {
        let products = self.products()?;
        let published = products.product(request, kind)?.ok_or_else(|| {
            crate::Error::Invalid(format!(
                "request {} has no {} product: the call that reads it is served only after the \
                 call that writes it, and nothing of this request wrote one",
                request.request_id.0,
                kind.name()
            ))
        })?;
        let bytes = usize::try_from(published.bytes)
            .map_err(|_| crate::Error::Invalid("a product is larger than memory".into()))?;
        let buffer = platform::PinnedBuffer::new(self.device, bytes)?;
        if !products.stage(request, kind, &buffer, &self.stream)? {
            return Err(crate::Error::Invalid(format!(
                "request {} lost its {} product between being asked for and being copied",
                request.request_id.0,
                kind.name()
            )));
        }
        Ok(buffer)
    }

    /// Performs the staged stage on the media lane, answering the call's completion.
    fn perform(&mut self, call: CallId) -> Result<BatchOutput> {
        let position = self
            .pending
            .iter()
            .position(|prepared| prepared.call == call)
            .ok_or_else(|| {
                crate::Error::Invariant(format!(
                    "the media stage was asked to serve call {} with no stage staged for it: a host \
                     job is returned by `submit` and served once, in that order",
                    call.0
                ))
            })?;
        let prepared = self.pending.remove(position);
        // A trajectory's opening has no bytes to encode: the mechanism's call is answered here, so
        // the request's first denoising step follows it as the graph states.
        if matches!(prepared.stage, Stage::Open) {
            return Ok(BatchOutput {
                calls: vec![CallOutput::completed(call, Vec::new(), RowState::default())],
                timing: ipc::TimingCounters::default(),
            });
        }
        let regions = prepared
            .read
            .as_ref()
            .map(|read| Arc::clone(&read.regions));
        let mut work = MediaWork::new(
            Arc::clone(&self.output),
            prepared.request,
            prepared.units,
            prepared.stage,
            prepared.input,
        );
        work.read = prepared.read;
        let started = Instant::now();
        let task = self.lane.admit(work)?;
        let answered = match task.wait() {
            Outcome::Success(bytes) => bytes,
            Outcome::Failed(error) => {
                return Err(crate::Error::Invalid(format!(
                    "request {}'s {:?} stage failed: {error}",
                    prepared.request.request_id.0, prepared.stage
                )));
            }
            Outcome::Cancelled => {
                return Err(crate::Error::State(
                    "a media stage was cancelled before it ran: the lane withdrew work the \
                     request's artifact cannot be assembled without",
                ));
            }
        };
        if let Some(regions) = regions {
            self.publish_regions(prepared.request, regions)?;
        }
        // The artifact is assembled, so nothing reads the request's conditions any more.
        if matches!(prepared.stage, Stage::Mux) {
            self.conditions.remove(&prepared.request);
        }
        let mut answer = CallOutput::completed(call, Vec::new(), RowState::default());
        if let Some(bytes) = answered.as_ref() {
            // The mux ends the request: its bytes are the artifact, published into a shared memory
            // object the receiving process claims by name and unlinks on receipt.
            let locator = ipc::MediaSource::publish(bytes)
                .map_err(|error| {
                    crate::Error::Invalid(format!(
                        "request {}'s artifact could not be published: {error}",
                        prepared.request.request_id.0
                    ))
                })?
                .into_locator();
            answer.media_output = Some(ipc::MediaOutput {
                handle: ipc::ArtifactHandle::PosixShm { name: locator.name },
                bytes: locator.bytes,
            });
            // The artifact holds what the request's products were for, so their life ends here: a
            // decoder's column is refused to the next request until they do.
            self.products()?.forget(prepared.request);
        }
        Ok(BatchOutput {
            calls: vec![answer],
            timing: ipc::TimingCounters {
                host_us: started.elapsed().as_micros() as u64,
                ..Default::default()
            },
        })
    }
}

/// Stages every request of one host-call batch, in the batch's order.
///
/// A batch holds one call per request, and the requests need not be at the same place in their
/// timelines: each request states its own canvas, its own frame count and its own unit, so the
/// entry a request takes is its own. Answering the first and leaving the
/// rest would complete a call whose stage never ran - the mux of that request would then wait for
/// units nothing encoded.
fn stage_batch(
    computation: crate::plan::Computation,
    requests: &[crate::batch::StagedRequest],
    units_of: &dyn Fn(u32) -> Result<Vec<u32>>,
) -> Result<Vec<StagedCall>> {
    let mut staged_calls = Vec::with_capacity(requests.len());
    for request in requests {
        let units = units_of(request.frames)?;
        let (stage, product) = staged(
            computation,
            (request.width, request.height),
            &units,
            request.unit as usize,
            request.handed,
        )?;
        staged_calls.push(StagedCall {
            request: request.request,
            units,
            stage,
            product,
        });
    }
    Ok(staged_calls)
}

/// One request of a host-call batch, staged: where in its timeline the stage works, and the product
/// it reads.
struct StagedCall {
    /// The request whose assembly the stage appends to.
    request: RequestKey,
    /// That request's timeline, as its schedule states it.
    units: Vec<u32>,
    /// The stage and the extent it works at.
    stage: Stage,
    /// The product the stage reads, and nothing for the mux.
    product: Option<Product>,
}

/// Which stage a host call is, and which product it reads.
///
/// The three calls differ in exactly these two facts, and stating them in one place is what keeps
/// the runner's own dispatch to the protocol: a video unit reads the postprocessor's raster, an
/// audio encode reads the decoder's window, and the mux reads only what the two encodes wrote.
fn staged(
    computation: crate::plan::Computation,
    canvas: (u32, u32),
    units: &[u32],
    index: usize,
    handed: u32,
) -> Result<(Stage, Option<Product>)> {
    let variant = computation.variant().ok_or_else(|| {
        crate::Error::Invalid(format!(
            "a {:?} batch is a computation no variant states",
            computation.call
        ))
    })?;
    match variant {
        // A video unit's frames are the schedule's entry for the place this unit takes in the
        // timeline, and its raster is the request's canvas - the shape the postprocessor wrote and
        // the encoder reads. Neither is an extent the mechanism states: a host call is sized by the
        // request, so a batch that named a canvas would be a second statement of it.
        Variant::VideoEncode => {
            let frames = units.get(index).copied().ok_or_else(|| {
                crate::Error::Invalid(format!(
                    "request's unit {index} is not one of the {} unit(s) its timeline holds: the \
                     encodes are appended in the schedule's order",
                    units.len()
                ))
            })?;
            let (width, height) = canvas;
            Ok((
                Stage::Unit {
                    index: index as u32,
                    frames,
                    height,
                    width,
                },
                Some(Product::Video(index as u32)),
            ))
        }
        Variant::AudioEncode => Ok((Stage::Track, Some(Product::Track))),
        Variant::Mux if handed > 0 => Ok((Stage::Hand, None)),
        Variant::Mux => Ok((Stage::Mux, None)),
        // A trajectory's opening computes nothing, and neither does an encoder whose work the call
        // that consumes it runs.
        Variant::LatentPrepare | Variant::ConditionEncode => Ok((Stage::Open, None)),
        Variant::MediaRead => Ok((Stage::Read, None)),
        other => Err(crate::Error::Invalid(format!(
            "the media stage serves the artifact's three calls and was asked for a {other:?} batch"
        ))),
    }
}

impl Runner for MediaStage {
    fn phases(&self) -> &[ipc::PhaseId] {
        &self.phases
    }

    /// A host call reads products through the store, in order after their writers, so it runs beside
    /// a clip the device is still decoding.
    fn overlaps_replays(&self) -> bool {
        true
    }

    fn stream(&self) -> Option<&platform::Stream> {
        Some(&self.stream)
    }

    /// One batch is one host call: stage the product it reads and hand the device thread the job
    /// that performs it.
    ///
    /// The refusal for a batch of another variant is by name: this runner serves the artifact's
    /// three calls and nothing else, and a media call it does not serve reaching it would otherwise
    /// be answered with an empty completion the mechanism reads as success.
    fn submit(&mut self, batch: Batch) -> Result<Jobs> {
        let media = batch.media.as_ref().ok_or_else(|| {
            crate::Error::Invalid(format!(
                "a batch of phase {} reached the media stage with no media columns: the three host \
                 calls are media computations and carry a staged request",
                batch.phase.0
            ))
        })?;
        media.check()?;
        let computation = media.computation;
        let requests = (0..batch.calls.len())
            .map(|index| media.request(index))
            .collect::<Result<Vec<_>>>()?;
        let staged_calls = stage_batch(computation, &requests, &|frames| (self.units_of)(frames))?;
        let mut prepared = Vec::with_capacity(staged_calls.len());
        let mut jobs = Vec::with_capacity(staged_calls.len());
        for ((call, staged_call), request) in batch
            .calls
            .iter()
            .copied()
            .zip(staged_calls)
            .zip(requests.iter())
        {
            let input = match staged_call.product {
                Some(kind) => Some(self.stage_product(staged_call.request, kind)?),
                None => None,
            };
            let read = match staged_call.stage {
                Stage::Read => Some(self.prepare_read(request)?),
                _ => None,
            };
            prepared.push(Prepared {
                call,
                request: staged_call.request,
                units: staged_call.units,
                stage: staged_call.stage,
                input,
                read,
            });
            jobs.push(Job::Host {
                runner: self.id,
                call,
            });
        }
        self.pending = prepared;
        Ok(Jobs::new(jobs))
    }

    fn serve(&mut self, job: &Job) -> Result<BatchOutput> {
        match job {
            Job::Host { call, .. } => self.perform(*call),
            other => Err(crate::Error::Invariant(format!(
                "the media stage serves host calls and the device asked it for {other:?}"
            ))),
        }
    }

    /// The three host calls this runner serves, as the plan states them.
    ///
    /// A host call captures no graph, so its bucket states only which device serves it. Stating
    /// them here is what makes the mechanism's question answerable: `Executor::media` answers from
    /// the plan's buckets, and a device that listed none would refuse every encode of the artifact
    /// with "the executor does not serve this computation".
    fn media_plan(&self) -> Vec<crate::plan::Bucket> {
        [
            Variant::VideoEncode,
            Variant::AudioEncode,
            Variant::Mux,
            Variant::LatentPrepare,
            Variant::MediaRead,
            Variant::ConditionEncode,
        ]
            .into_iter()
            .map(crate::plan::Bucket::host)
            .collect()
    }

    /// The store the request's products are read from (4.5's executor-owned storage).
    fn attach_products(&mut self, products: Arc<dyn Products>) {
        self.products = Some(products);
    }

    /// Nothing to capture: this runner owns no graph. The three calls a media request ends with
    /// are host work, so a generation changes under them without changing anything here - the
    /// encoder and the muxer belong to the deployment, not to a program generation.
    fn warmup(&mut self) -> Result<Jobs> {
        Ok(Jobs::default())
    }

    /// Nothing to prepare, for the same reason: no graph of this runner reads a program.
    fn precopy(&mut self, _next: &Arc<ops::Generation>) -> Result<Jobs> {
        Ok(Jobs::default())
    }

    /// Nothing to switch: the host work reads no kernel, so a generation boundary is empty here.
    fn switchover(&mut self, _generation: u64) -> Result<()> {
        Ok(())
    }

    /// Nothing to revert, for the same reason.
    fn revert(&mut self, _generation: u64) {}

    /// Nothing to release: this runner holds no graph that references a generation.
    fn purge(&mut self, _generation: u64) {}

    /// Nothing to discard, for the same reason.
    fn discard(&mut self, _generation: u64) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Frames reach the resize as planes, one frame after another: an interleaved raster read as
    /// planes would resize a mixture of the channels, and every shape would agree.
    #[test]
    fn interleaved_frames_become_planes_one_frame_after_another() {
        // Two frames of one row of two pixels.
        let rgb = [1_u8, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16];
        assert_eq!(
            planes_of(&rgb, 2, 1, 2),
            [1, 4, 2, 5, 3, 6, 11, 14, 12, 15, 13, 16]
        );
    }

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
        // The ranks finish a timeline's units in any order, and the assembly takes them in its own.
        for (index, frames) in units.iter().copied().enumerate().rev() {
            let pixels = vec![0x40_u8; frames as usize * height as usize * width as usize * 3];
            unit_bytes += pixels.len();
            output
                .encode_clip(request, &units, index, (frames, height, width), &pixels)
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

    /// A track that arrives before the first unit waits for the unit that opens the assembly: the
    /// two encode beside each other on different ranks.
    #[test]
    fn a_track_that_arrives_first_joins_the_assembly_its_units_open() {
        let output = Output::from_env(Rates {
            frame_rate: 24,
            audio_rate: 32_000,
        })
        .expect("the runtime provides libx264 and aac");
        let request = RequestKey::new(0, ipc::RequestId(8), 0);
        let units = [17_u32, 22];
        let (height, width) = (16_u32, 16_u32);
        let samples = media::codec::audio_samples(
            units.iter().map(|&frames| frames as usize).sum::<usize>(),
            24,
            32_000,
        );
        let pcm = vec![0_u8; samples * 2 * size_of::<i16>()];
        output
            .encode_track(request, &pcm, samples)
            .expect("a track encodes before any unit");
        for (index, frames) in units.iter().copied().enumerate() {
            let pixels = vec![0x40_u8; frames as usize * height as usize * width as usize * 3];
            output
                .encode_clip(request, &units, index, (frames, height, width), &pixels)
                .expect("a unit encodes");
        }
        let mp4 = output.finish(request).expect("the assembly ends");
        assert_eq!(&mp4[4..8], b"ftyp", "an MP4 begins with its file type box");
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
            .encode_clip(request, &[17, 22], 0, (17, 2, 2), &[0_u8; 16])
            .expect_err("a short clip");
        let message = format!("{error}");
        assert!(
            message.contains("204") && message.contains("16"),
            "the refusal names both byte counts: {message}"
        );
    }

    /// A host call names the stage it is and the product it reads, and nothing else is served.
    ///
    /// The two facts are what the runner owes the protocol: a video encode works at one unit of the
    /// request's timeline and reads the postprocessor's frames, an audio encode reads the decoder's
    /// track, and the mux reads neither - its inputs are the two encodes the deployment's output
    /// already holds. A call of another class reaching the stage would otherwise be answered with an
    /// empty completion the mechanism reads as success.
    /// A batch of two requests stages both, each at its own timeline's entry.
    ///
    /// The mechanism batches media requests as they arrive, and a host bucket states no bound, so a
    /// batch of two encodes is two requests whose artifacts are being assembled at once. Each has
    /// its own canvas, its own frame count and its own unit: staging the first
    /// and answering the batch would leave the second's mux waiting for a unit nothing encoded.
    #[test]
    fn a_batch_of_two_requests_stages_both_at_their_own_entries() {
        let requests = [
            crate::batch::StagedRequest {
                unit: 0,
                request: ipc::RequestKey::new(1, ipc::RequestId(7), 0),
                conditioning: Vec::new(),
                step: 0,
                steps: 8,
                seed: 11,
                width: 1344,
                height: 768,
                frames: 56,
                handed: 0,
            },
            // The second request's first unit is already through; the first request's is not.
            crate::batch::StagedRequest {
                unit: 1,
                request: ipc::RequestKey::new(1, ipc::RequestId(8), 0),
                conditioning: Vec::new(),
                step: 0,
                steps: 8,
                seed: 12,
                width: 768,
                height: 768,
                frames: 39,
                handed: 0,
            },
        ];
        let video = crate::plan::Computation {
            call: ipc::MediaCall::VideoEncoding,
            rows: 2,
            tokens: 0,
            extent: crate::plan::Extent::None,
        };
        let units_of = |frames: u32| match frames {
            56 => Ok(vec![22_u32, 22, 12]),
            39 => Ok(vec![22_u32, 17]),
            other => Err(crate::Error::Invalid(format!("no schedule for {other}"))),
        };

        let staged_calls = stage_batch(video, &requests, &units_of).expect("both");
        assert_eq!(
            staged_calls
                .iter()
                .map(|staged_call| (staged_call.request, staged_call.stage))
                .collect::<Vec<_>>(),
            vec![
                (
                    requests[0].request,
                    Stage::Unit {
                        index: 0,
                        frames: 22,
                        height: 768,
                        width: 1344,
                    },
                ),
                (
                    requests[1].request,
                    Stage::Unit {
                        index: 1,
                        frames: 17,
                        height: 768,
                        width: 768,
                    },
                ),
            ],
            "the first request's first unit, and the second's second"
        );
        assert!(staged_calls
            .iter()
            .all(|staged_call| matches!(staged_call.product, Some(Product::Video(_)))));
    }

    #[test]
    fn a_host_call_states_the_stage_and_the_product_it_reads() {
        let video = crate::plan::Computation {
            call: ipc::MediaCall::VideoEncoding,
            rows: 1,
            tokens: 0,
            extent: crate::plan::Extent::None,
        };
        let units = [17_u32, 22];

        // The first unit of the timeline is the first entry of the schedule, at the request's own
        // canvas: the batch states no extent, and nothing about the raster is re-derived here.
        assert_eq!(
            staged(video, (1344, 768), &units, 0, 0).expect("a video encode"),
            (
                Stage::Unit {
                    index: 0,
                    frames: 17,
                    height: 768,
                    width: 1344,
                },
                Some(Product::Video(0)),
            )
        );
        assert_eq!(
            staged(video, (1344, 768), &units, 1, 0).expect("the timeline's second unit"),
            (
                Stage::Unit {
                    index: 1,
                    frames: 22,
                    height: 768,
                    width: 1344,
                },
                Some(Product::Video(1)),
            )
        );

        // Only the muxing call that hands over nothing finishes the artifact.
        let mux = crate::plan::Computation {
            call: ipc::MediaCall::Muxing,
            ..video
        };
        assert_eq!(
            staged(mux, (1344, 768), &units, 0, 2).expect("a round handed over"),
            (Stage::Hand, None)
        );
        assert_eq!(
            staged(mux, (1344, 768), &units, 0, 0).expect("the final call"),
            (Stage::Mux, None)
        );

        let audio = crate::plan::Computation {
            call: ipc::MediaCall::AudioEncoding,
            ..video
        };
        assert_eq!(
            staged(audio, (1344, 768), &units, 0, 0).expect("an audio encode"),
            (Stage::Track, Some(Product::Track))
        );

        let mux = crate::plan::Computation {
            call: ipc::MediaCall::Muxing,
            ..audio
        };
        assert_eq!(
            staged(mux, (1344, 768), &units, 0, 0).expect("a mux"),
            (Stage::Mux, None)
        );

        // A unit past the schedule's end is refused by name rather than given the last entry: two
        // encodes running as one is an assembly the muxer would refuse much later.
        let message = format!(
            "{}",
            staged(video, (1344, 768), &units, 2, 0).expect_err("no third unit")
        );
        assert!(
            message.contains("2 unit(s)"),
            "the refusal names the timeline's length: {message}"
        );

        // A device computation is another runner's: this one serves the artifact's three calls.
        let denoise = crate::plan::Computation {
            call: ipc::MediaCall::Denoising,
            ..video
        };
        let message = format!(
            "{}",
            staged(denoise, (1344, 768), &units, 0, 0).expect_err("a denoise")
        );
        assert!(
            message.contains("three calls"),
            "the refusal states what this runner serves: {message}"
        );
    }
}

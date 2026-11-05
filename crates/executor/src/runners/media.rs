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
use std::time::Instant;

use ipc::RequestKey;

use crate::Result;
use crate::batch::{Batch, BatchOutput, CallId, CallOutput, Job, Jobs, RowState};
use crate::host::{HostAction, Lane, Outcome};
use crate::plan::Variant;
use crate::runner::{Binding, Runner, RunnerId};
use crate::runners::products::{Product, Products};

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
    /// How many units each request's assembly holds, counted as they are appended.
    ///
    /// A unit's place in the timeline is what says how many frames it covers, and the mechanism's
    /// encode batch states no extent to carry it: the count is this output's own record of the
    /// appends, which is the same order the muxer checks against `units` when it finishes.
    appended: std::sync::Mutex<std::collections::HashMap<u64, usize>>,
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
            appended: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// How many units of `request`'s timeline this output has appended.
    ///
    /// The stage that encodes a unit needs the frames that unit covers, and the mechanism's batch
    /// states them nowhere: a media encode is sized by the raster and the assembly, never by an
    /// extent, so the place in the timeline is read from what this output has already appended.
    pub fn units_appended(&self, request: RequestKey) -> usize {
        crate::lock(&self.appended)
            .get(&request_id(request))
            .copied()
            .unwrap_or(0)
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
        // The append is what says this unit has its place: the next encode of this request reads
        // the schedule's following entry, so the count moves only after the encoder accepted it.
        *crate::lock(&self.appended)
            .entry(request_id(request))
            .or_insert(0) += 1;
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
        }
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
        let work = MediaWork::new(
            Arc::clone(&self.output),
            prepared.request,
            prepared.units,
            prepared.stage,
            prepared.input,
        );
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
/// timelines: each request states its own canvas, its own frame count and its own count of
/// appended units, so the entry a request takes is its own. Answering the first and leaving the
/// rest would complete a call whose stage never ran - the mux of that request would then wait for
/// units nothing encoded.
fn stage_batch(
    computation: crate::plan::Computation,
    requests: &[crate::batch::StagedRequest],
    units_of: &dyn Fn(u32) -> Result<Vec<u32>>,
    appended: &dyn Fn(RequestKey) -> usize,
) -> Result<Vec<StagedCall>> {
    let mut staged_calls = Vec::with_capacity(requests.len());
    for request in requests {
        let units = units_of(request.frames)?;
        let (stage, product) = staged(
            computation,
            (request.width, request.height),
            &units,
            appended(request.request),
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
                    frames,
                    height,
                    width,
                },
                Some(Product::Video),
            ))
        }
        Variant::AudioEncode => Ok((Stage::Track, Some(Product::Track))),
        Variant::Mux => Ok((Stage::Mux, None)),
        other => Err(crate::Error::Invalid(format!(
            "the media stage serves the artifact's three calls and was asked for a {other:?} batch"
        ))),
    }
}

impl Runner for MediaStage {
    fn phases(&self) -> &[ipc::PhaseId] {
        &self.phases
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
        let staged_calls = stage_batch(
            computation,
            &requests,
            &|frames| (self.units_of)(frames),
            &|request| self.output.units_appended(request),
        )?;
        let mut prepared = Vec::with_capacity(staged_calls.len());
        let mut jobs = Vec::with_capacity(staged_calls.len());
        for (call, staged_call) in batch.calls.iter().copied().zip(staged_calls) {
            let input = match staged_call.product {
                Some(kind) => Some(self.stage_product(staged_call.request, kind)?),
                None => None,
            };
            prepared.push(Prepared {
                call,
                request: staged_call.request,
                units: staged_call.units,
                stage: staged_call.stage,
                input,
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
        [Variant::VideoEncode, Variant::AudioEncode, Variant::Mux]
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
    /// its own canvas, its own frame count and its own count of appended units: staging the first
    /// and answering the batch would leave the second's mux waiting for a unit nothing encoded.
    #[test]
    fn a_batch_of_two_requests_stages_both_at_their_own_entries() {
        let requests = [
            crate::batch::StagedRequest {
                request: ipc::RequestKey::new(1, ipc::RequestId(7), 0),
                conditioning: Vec::new(),
                step: 0,
                steps: 8,
                seed: 11,
                width: 1344,
                height: 768,
                frames: 56,
            },
            crate::batch::StagedRequest {
                request: ipc::RequestKey::new(1, ipc::RequestId(8), 0),
                conditioning: Vec::new(),
                step: 0,
                steps: 8,
                seed: 12,
                width: 768,
                height: 768,
                frames: 39,
            },
        ];
        let video = crate::plan::Computation {
            call: ipc::MediaCall::VideoEncoding,
            rows: 2,
            tokens: 0,
            extent: crate::plan::Extent::None,
        };
        // The second request has already had one unit appended; the first has had none.
        let appended = |request: ipc::RequestKey| usize::from(request.request_id.0 == 8);
        let units_of = |frames: u32| match frames {
            56 => Ok(vec![22_u32, 22, 12]),
            39 => Ok(vec![22_u32, 17]),
            other => Err(crate::Error::Invalid(format!("no schedule for {other}"))),
        };

        let staged_calls = stage_batch(video, &requests, &units_of, &appended).expect("both");
        assert_eq!(
            staged_calls
                .iter()
                .map(|staged_call| (staged_call.request, staged_call.stage))
                .collect::<Vec<_>>(),
            vec![
                (
                    requests[0].request,
                    Stage::Unit {
                        frames: 22,
                        height: 768,
                        width: 1344,
                    },
                ),
                (
                    requests[1].request,
                    Stage::Unit {
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
            .all(|staged_call| staged_call.product == Some(Product::Video)));
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
            staged(video, (1344, 768), &units, 0).expect("a video encode"),
            (
                Stage::Unit {
                    frames: 17,
                    height: 768,
                    width: 1344,
                },
                Some(Product::Video),
            )
        );
        assert_eq!(
            staged(video, (1344, 768), &units, 1).expect("the timeline's second unit"),
            (
                Stage::Unit {
                    frames: 22,
                    height: 768,
                    width: 1344,
                },
                Some(Product::Video),
            )
        );

        let audio = crate::plan::Computation {
            call: ipc::MediaCall::AudioEncoding,
            ..video
        };
        assert_eq!(
            staged(audio, (1344, 768), &units, 0).expect("an audio encode"),
            (Stage::Track, Some(Product::Track))
        );

        let mux = crate::plan::Computation {
            call: ipc::MediaCall::Muxing,
            ..audio
        };
        assert_eq!(
            staged(mux, (1344, 768), &units, 0).expect("a mux"),
            (Stage::Mux, None)
        );

        // A unit past the schedule's end is refused by name rather than given the last entry: two
        // encodes running as one is an assembly the muxer would refuse much later.
        let message = format!(
            "{}",
            staged(video, (1344, 768), &units, 2).expect_err("no third unit")
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
            staged(denoise, (1344, 768), &units, 0).expect_err("a denoise")
        );
        assert!(
            message.contains("three calls"),
            "the refusal states what this runner serves: {message}"
        );
    }
}

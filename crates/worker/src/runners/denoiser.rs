//! `DenoiserRunner`: the runner of a diffusion model's denoiser (`nn::Denoiser`), which prepares
//! a video request's latents and advances them one denoising step per call.
//!
//! The runner holds its model through `nn::Denoiser` and names no concrete model. A media row
//! names the products its call reads and writes as positions in the request's product table
//! (`ipc::batch::MediaRow::results`); the runner tells them apart by the class whose entry the
//! rank's device block tables hold at each position, and finds their blocks in the product
//! buffer of the rank's [`BufferMemory`].
//!
//! ## The request's state in its slot
//!
//! A request's state lives from its latent preparation to its last step in its slot's row of the
//! rank's trajectory rows ([`BufferMemory::trajectory`]), laid out by the request's layout from
//! the row's first byte: its video latents `[channels, frames, rows, columns]` and its audio
//! latents `[channels, steps]` in the sample type, back to back, then from a unit boundary the
//! conditioning rows the denoiser reads (`[prefix rows, text width]`) and from the next the
//! encoder's rows the conditioner reads (`[prefix rows, width]`), both in the stack's type. The
//! latents lead, so the trajectory a `Save` copies is the row's first bytes
//! ([`trajectory_layout`]); a `Restore` may write a page past them, which the request's latent
//! preparation, which follows its restore, writes again.
//!
//! ## A call
//!
//! Every batch is one request's call; a batch with no row carries copies alone and returns no
//! call.
//!
//! - **A latent preparation** runs eagerly ([`Graphs::eager`]) over the slot's row: it copies the
//!   prompt's conditioning rows into the encoder's rows, and into the denoiser's when the two
//!   widths agree, the rest of both zero; lends the condition pixels and samples as the media host
//!   left them, each condition's after the ones before it in request order; draws each condition's
//!   anchoring noise and the request's noise; runs the model's `prepare_conditioning`; and starts
//!   the schedule (`prepare_latents`). A request whose row states a step past zero was restored
//!   from a stored trajectory at that step: its latents are kept, and only its conditioning is
//!   prepared again.
//! - **A denoising step** is replayed from its bucket's graph, which reads the runner's state
//!   buffer at fixed addresses: the slot's latents and conditioning are copied into the buffer
//!   before the replay and the latents back after it, on the runner's stream. The last step also
//!   copies the latents into the video latents and audio latents blocks its row names, which the
//!   decoders read as `[channels, frames · rows · columns]` and as the track's elements in the
//!   sample type, their `latent_type`. The first rank of the denoising group
//!   ([`DenoiserConfig::first`]) copies them on to the backing ranks of its process
//!   ([`BufferMemory::share`]).
//!
//! ## Buckets
//!
//! One bucket per distinct layout of each served request shape, from the deployment's
//! candidates (`nn::family::VideoModel::candidates`) in their order: the model's layout of a
//! candidate is the bucket's extent, and candidates its padding collapses onto one layout share
//! the bucket of the first, which is the request its graph is captured from. A step's bucket is
//! captured; a preparation's bucket of the same extent is profiled and run eagerly
//! ([`Runner::eager_buckets`]), and its profile opens the candidate and each condition set a
//! request of it may open with, after the model's initialization, so the runner's region holds
//! the largest opening and every generation resolves their kernels.
//!
//! ## Save and restore
//!
//! [`Runner::copy`] issues the copies of a batch through the rank's [`BufferMemory`]. Every rank
//! of every parallel configuration holds a request's whole samples in its slot's row, laid out by
//! the request's layout alone, so a saved trajectory has one host copy: a `Save` copies the
//! row of the group's first rank ([`DenoiserConfig::first`]) into the part of the host latent
//! region the rank copies with ([`BufferMemory::part_holder`]), and a `Restore` copies it back
//! into the row of every rank of the restoring group, whichever group and layout saved it. The
//! class of stored trajectories states its tensors and its encoding over [`trajectory_layout`]
//! of the largest layout, once ([`DenoiserRunner::trajectory_digests`]). A `Restore` whose copy
//! names another encoding of the same tensors was saved by a deployment whose largest layout is
//! another; the request's trajectory is its own layout's bytes under both, so it is laid out
//! through the copy manifest from that same part as the bytes they are
//! ([`BufferMemory::restore`]).

use std::cell::RefCell;
use std::sync::Arc;
use std::time::Instant;

use ::storage::manifest::{Encoding, Field, Logical, Segment, Span, StorageLayout};
use ::storage::runtime;
use ipc::batch::{
    Batch, BatchOutput, CallOutput, Extent, MediaRow, Phase, Shape, TimingCounters, Variant,
};
use nn::{ConditionPiece, ConditionRole, ConditionShape, Conditioning, Denoiser};
use nn::{LatentShape, MediaInput, Step, VideoRequest};
use platform::{Event, EventFlags, PitchedBatch, SourceOrder, Stream};
use proto::video::Layout;
use tensor::{DType, Tensor};

use super::decoder::{elements, pitched, voxels};
use super::{address, element_bytes, performed, settle};
use crate::buckets::{Bucket, Buckets};
use crate::graphs::Graphs;
use crate::memory::allocation::Storage;
use crate::memory::buffers::{BufferMemory, UNIT_BYTES};
use crate::memory::latent::restore::Stored;
use crate::memory::lend;
use crate::memory::tables::{BlockCopy, BlockTables};
use crate::runner::Runner;
use crate::{Error, Result};

/// The element type the media host lays a request's condition pixels and samples out in: the
/// form `nn::MediaFormat` states for the encoders the conditioner runs.
pub const CONDITION_MEDIA: DType = DType::F32;

/// The products a call reads and writes, as indices into [`DenoiserConfig::classes`].
const CONDITIONING: usize = 0;
const PIXELS: usize = 1;
const SAMPLES: usize = 2;
const VIDEO: usize = 3;
const AUDIO: usize = 4;

/// The products' names, in the order of [`DenoiserConfig::classes`].
const NAMES: [&str; 5] = [
    "conditioning",
    "condition pixels",
    "condition samples",
    "video latents",
    "audio latents",
];

/// A served bucket: the step's bucket, its layout, the candidate it is captured from, and the
/// condition sets a request of it may open with.
type Served = (Bucket, Layout, VideoRequest, Vec<Vec<ConditionShape>>);

/// A request's state over the parts of its layout: its video latents, its audio latents, the
/// conditioning rows the denoiser reads and the encoder's rows the conditioner reads.
type State = (Tensor, Option<Tensor>, Tensor, Tensor);

/// The latent preparation being served: its row, the request it states, the model's pieces of
/// its conditions, and the conditioning, condition pixels and condition samples it reads, lent
/// from their blocks.
type Opening = (
    MediaRow,
    VideoRequest,
    Vec<ConditionPiece>,
    [Option<Tensor>; 3],
);

/// What a step moves around its replay: the slot's state into the state buffer before it, the
/// latents back and, on the last step, into their blocks after it, and the blocks it wrote, by
/// class and position.
type Moves = (PitchedBatch, PitchedBatch, Vec<(u8, u32)>);

/// What a deployment fixes of a denoiser runner. Immutable once the runner is built.
#[derive(Clone, Debug)]
pub struct DenoiserConfig {
    /// The phases the runner serves: a video request's latent preparation and its steps.
    pub phases: Vec<Phase>,
    /// The requests the deployment serves (`nn::family::VideoModel::candidates`): one per layout a
    /// request may take, each with the condition sets a request of it may open with.
    pub candidates: Vec<(VideoRequest, Vec<Vec<ConditionShape>>)>,
    /// The element type of the conditioning the text encoder writes, which the stack reads.
    pub stack: DType,
    /// The element type of the trajectory and its noise (`VideoModel::sample_type`).
    pub sample: DType,
    /// The width of the conditioning rows the text encoder writes (`VideoModel::text_width`).
    pub width: u32,
    /// The classes of the products: the conditioning, the condition pixels and the condition
    /// samples a preparation reads, then the video latents and the audio latents the last step
    /// writes.
    pub classes: [u8; 5],
    /// The ranks of the denoising group, each holding the whole samples of its requests.
    pub ranks: u32,
    /// The bytes of a unit of a part of the host latent region (`BankPages::page_bytes`).
    pub page_bytes: u64,
    /// Whether this rank is the first of its denoising group: it copies the latents the last
    /// step wrote to the backing ranks of its process, and it alone writes a saved trajectory's
    /// host copy, which every rank of the group holds whole.
    pub first: bool,
}

/// The runner of a denoiser (module documentation).
pub struct DenoiserRunner {
    model: Arc<dyn Denoiser + Send + Sync>,
    config: DenoiserConfig,
    graphs: Graphs,
    /// The rank's media memory, which every media runner of the rank shares.
    memory: Arc<BufferMemory>,
    /// The stream the graphs replay on; `None` for graphs that hold no device.
    stream: Option<Stream>,
    /// Recorded behind a batch's last copy: what `execute` waits for.
    done: Option<Event>,
    /// The served buckets, in the candidates' order.
    served: Vec<Served>,
    /// The steps' buckets.
    buckets: Buckets,
    /// The width of the conditioning rows the denoiser reads.
    text_width: u32,
    /// The state buffer the captured steps read, `bytes` long, laid out as a slot's row is;
    /// `None` for graphs that hold no device.
    state: Option<Arc<Storage>>,
    bytes: u64,
    /// The digests of the trajectory class's tensors and of its encoding.
    tensors: [u8; 32],
    encoding: [u8; 32],
    /// The step or the latent preparation being served, which the forward reads; `None` while
    /// a profile run or a capture computes over a candidate.
    current_step: RefCell<Option<(Step, VideoRequest)>>,
    current_opening: RefCell<Option<Opening>>,
}

impl DenoiserRunner {
    /// A runner of `model` under `config`, over `memory`, whose trajectory rows are
    /// [`DenoiserRunner::row_bytes`] long.
    ///
    /// - `graphs` are made for no input tensors: a step's host values are its forward's.
    /// - `stream` is the stream `graphs` replay on, on which the runner issues its copies; `None`
    ///   for graphs that hold no device, whose runner computes over meta tensors.
    ///
    /// The model's tables are filled here on a device ([`Denoiser::initialize`]).
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for no candidate, no rank, a stream without a device or the reverse;
    /// what the model refuses of a candidate; the driver's refusal of the state buffer.
    pub fn new(
        model: Arc<dyn Denoiser + Send + Sync>,
        config: DenoiserConfig,
        graphs: Graphs,
        stream: Option<Stream>,
        memory: Arc<BufferMemory>,
    ) -> Result<Self> {
        if graphs.enter().is_ok() != stream.is_some() || config.ranks == 0 {
            return Err(Error::Invalid(
                "a denoiser runner takes a group of ranks, and the stream of its graphs or none \
                 for graphs that hold no device"
                    .into(),
            ));
        }
        let served = served(&*model, &config.candidates)?;
        let text_width = model.text_width().unwrap_or(config.width);
        let (bytes, largest) = extents(&served, &config, text_width)?;

        // The trajectory class, by the largest trajectory a row holds: one host copy whatever
        // the group's width, since every rank of a group holds the whole samples.
        let laid = trajectory_layout(largest, config.sample)?;
        let (tensors, encoding) = runtime::encoding(std::slice::from_ref(&laid))
            .map_err(|error| Error::Invalid(format!("a trajectory's layout: {error}")))?;

        // The state buffer holds defined values for the capture's first forward.
        let state = match &stream {
            Some(stream) => {
                let length = usize::try_from(bytes)
                    .map_err(|_| Error::Invalid("the denoiser's state exceeds memory".into()))?;
                let storage = Storage::linear(stream.device(), length)?;

                // SAFETY: the range is the whole allocation, which this runner keeps for as long
                // as the graphs captured over it exist.
                unsafe { platform::memset_d8(storage.address(), 0, length, stream) }?;
                Some(storage)
            }
            None => None,
        };

        let buckets = Buckets::new(served.iter().map(|(bucket, ..)| *bucket));
        let runner = Self {
            done: stream.as_ref().map(|_| Event::new(EventFlags::default())),
            model,
            config,
            graphs,
            memory,
            stream,
            served,
            buckets,
            text_width,
            state,
            bytes,
            tensors,
            encoding,
            current_step: RefCell::new(None),
            current_opening: RefCell::new(None),
        };
        runner.initialize()?;
        Ok(runner)
    }

    /// The bytes of one slot's row in the rank's media memory (`BufferMemory::open`), and of the
    /// runner's state buffer: the state of the largest request `config` serves.
    ///
    /// # Errors
    ///
    /// No candidate, what the model refuses of one, or an element type with no whole bytes.
    pub fn row_bytes(model: &dyn Denoiser, config: &DenoiserConfig) -> Result<u64> {
        let text_width = model.text_width().unwrap_or(config.width);
        let served = served(model, &config.candidates)?;

        extents(&served, config, text_width).map(|(bytes, _)| bytes)
    }

    /// The bytes of device memory the runner holds beside its graphs' region: its state buffer.
    pub const fn device_bytes(&self) -> u64 {
        self.bytes
    }

    /// The trajectory class's `tensors` and `encoding` digests (`storage::runtime::Class`): the
    /// largest trajectory's [`trajectory_layout`], its one host copy.
    pub fn trajectory_digests(&self) -> ([u8; 32], [u8; 32]) {
        (self.tensors, self.encoding)
    }

    /// Per captured shape, as `Capacity::media_bytes` states it: the request shape it serves, the
    /// bytes of a request's trajectory, and of its video latents and audio latents blocks.
    pub fn media_shapes(&self) -> Result<Vec<(Shape, u64, [u64; 2])>> {
        let mut shapes = Vec::with_capacity(self.served.len());
        for (bucket, layout, ..) in &self.served {
            let (frames, width, height) = bucket.serves.ok_or_else(|| {
                Error::Invariant(format!("bucket {bucket} serves no request shape"))
            })?;
            let [video, audio, ..] = self.parts(layout)?;
            let shape = Shape {
                rows: bucket.rows,
                tokens: 0,
                variant: Variant::Denoise,
                extent: Extent::Request {
                    frames,
                    width,
                    height,
                    text_rows: layout.text_rows,
                    condition_rows: layout.condition_rows,
                },
            };
            shapes.push((shape, video.1 + audio.1, [video.1, audio.1]));
        }
        Ok(shapes)
    }

    /// Fills the model's tables on the device, under the launching context, and waits for them.
    fn initialize(&self) -> Result<()> {
        let Some(stream) = &self.stream else {
            return Ok(());
        };

        {
            let _guard = self.graphs.enter()?;
            self.model.initialize()?;
        }
        Ok(stream.synchronize()?)
    }

    /// The parts of a request of `layout` in a slot's row.
    fn parts(&self, layout: &Layout) -> Result<[(u64, u64); 4]> {
        parts(layout, &self.config, self.text_width)
    }

    /// The served bucket of `bucket`'s extent, a step's or a preparation's.
    fn served(&self, bucket: &Bucket) -> Result<&Served> {
        let step = Bucket {
            variant: Variant::Denoise,
            ..*bucket
        };
        self.served
            .iter()
            .find(|(held, ..)| *held == step)
            .ok_or_else(|| Error::Invariant(format!("bucket {bucket} is none the runner states")))
    }

    /// The tensor of `shape` elements of `dtype` from byte `offset` of the state buffer.
    fn buffer(&self, offset: u64, shape: &[usize], dtype: DType) -> Result<Tensor> {
        let count: usize = shape.iter().product();
        if offset + count as u64 * element_bytes(dtype)? > self.bytes {
            return Err(Error::Invariant(format!(
                "{shape:?} {dtype:?} from byte {offset} lies past the denoiser's state"
            )));
        }
        let base = self.state.as_ref().map_or(0, |storage| storage.address());

        // SAFETY: the tensor lies inside the state buffer, which this runner owns for as long as
        // its graphs exist; only the copies it issues on its stream and its forwards write it.
        unsafe {
            lend(
                base + offset,
                shape,
                dtype,
                self.stream.as_ref().map(Stream::device),
            )
        }
    }

    /// A request's state over `layout`: in `slot`'s row, or for no slot in the state buffer.
    fn state(&self, layout: &Layout, slot: Option<u32>) -> Result<State> {
        let [video, audio, text, source] = self.parts(layout)?;
        let (sample, stack) = (self.config.sample, self.config.stack);
        let lent = |(offset, _): (u64, u64), shape: &[usize], dtype| match slot {
            Some(slot) => self.memory.trajectory(slot, offset, shape, dtype),
            None => self.buffer(offset, shape, dtype),
        };

        let shape = layout.video;
        let dims = [shape.channels, shape.frames, shape.height, shape.width].map(|d| d as usize);
        let rows = nn::capability::prefix_rows(layout) as usize;
        let track = match layout.audio {
            Some(track) => {
                let dims = [track.channels as usize, voxels(&track) as usize];
                Some(lent(audio, &dims, sample)?)
            }
            None => None,
        };
        Ok((
            lent(video, &dims, sample)?,
            track,
            lent(text, &[rows, self.text_width as usize], stack)?,
            lent(source, &[rows, self.config.width as usize], stack)?,
        ))
    }

    /// The bucket of `batch`, a step's or a preparation's of the extent its shape states, and
    /// its one row.
    fn call<'a>(&self, batch: &'a Batch) -> Result<(Bucket, &'a MediaRow)> {
        batch.check()?;
        let variant = batch.shape.variant;
        if !matches!(variant, Variant::Denoise | Variant::LatentPrepare) {
            return Err(Error::Invalid(format!(
                "a denoiser runner serves no {} batch",
                variant.name()
            )));
        }
        let step = Shape {
            variant: Variant::Denoise,
            ..batch.shape
        };
        let bucket = self.buckets.select(&step).ok_or_else(|| {
            Error::Invalid(format!("no bucket of the denoiser holds {}", batch.shape))
        })?;
        let [row] = batch.media.as_slice() else {
            return Err(Error::Invalid(format!(
                "batch {} carries {} requests in a bucket of one",
                batch.id.0,
                batch.media.len()
            )));
        };
        Ok((Bucket { variant, ..bucket }, row))
    }

    /// The position of each product of `wanted` among `results`, by the class whose entry the
    /// tables hold at each position.
    fn named<const N: usize>(
        &self,
        tables: &BlockTables,
        row: &MediaRow,
        results: &[u32],
        wanted: [usize; N],
    ) -> Result<[Option<u32>; N]> {
        let mut found = [None; N];
        for &position in results {
            let held = |kind: usize| {
                let class = self.config.classes[kind];
                !tables.runs(row.slot, class, position).is_empty()
            };
            let index = wanted.iter().position(|&kind| held(kind)).ok_or_else(|| {
                Error::Invalid(format!(
                    "request {} names result {position}, which the call neither reads nor writes",
                    row.request.0
                ))
            })?;
            found[index] = Some(position);
        }
        Ok(found)
    }

    /// The latent preparation of `row` in `bucket`: the request it states and the products it
    /// reads, lent from their blocks (module documentation).
    fn opening(&self, bucket: &Bucket, row: &MediaRow, tables: &BlockTables) -> Result<Opening> {
        let refused = |problem: String| {
            Error::Invalid(format!(
                "the latent preparation of request {} {problem}",
                row.request.0
            ))
        };
        let (_, _, warm, _) = self.served(bucket)?;
        let request = stated(warm, &row.conditioning)?;
        let steps = self.model.schedule(&request)?.len();
        if row.reads as usize != row.results.len() || row.step as usize >= steps {
            return Err(refused(format!(
                "writes a result, or starts at step {} of {steps}",
                row.step
            )));
        }

        let read = self.named(tables, row, &row.results, [CONDITIONING, PIXELS, SAMPLES])?;
        let pieces = self.model.condition_pieces(&request)?;
        let total = |extent: fn(&ConditionPiece) -> Option<&nn::Shape>| -> usize {
            pieces
                .iter()
                .filter_map(extent)
                .map(nn::Shape::elem_count)
                .sum()
        };
        let lent = |kind: usize, shape: &[usize], dtype: DType| -> Result<Tensor> {
            let position =
                read[kind].ok_or_else(|| refused(format!("names no {} block", NAMES[kind])))?;
            let entry = (row.slot, self.config.classes[kind], position);

            self.memory.lend(tables, entry, false, 0, shape, dtype)
        };
        // A condition medium is read only by a request whose pieces hold some of it.
        let media = |kind: usize, total: usize| match total {
            0 => Ok(None),
            total => lent(kind, &[total], CONDITION_MEDIA).map(Some),
        };

        let prompt = [request.text_tokens as usize, self.config.width as usize];
        let products = [
            Some(lent(CONDITIONING, &prompt, self.config.stack)?),
            media(PIXELS, total(|piece| piece.pixels.as_ref()))?,
            media(SAMPLES, total(|piece| piece.track.as_ref()))?,
        ];
        Ok((row.clone(), request, pieces, products))
    }

    /// Opens `request` of `layout` over `state` (module documentation): its conditioning from the
    /// `products` it reads, its noise from `seed` unless it was `restored`, and the model's
    /// preparation of both.
    fn open(
        &self,
        layout: &Layout,
        (request, pieces): (&VideoRequest, &[ConditionPiece]),
        (seed, restored): (u64, bool),
        (video, audio, text, source): State,
        [conditioning, pixels, samples]: [Option<Tensor>; 3],
    ) -> Result<()> {
        let conditioning = conditioning
            .ok_or_else(|| Error::Invariant("an opening reads its conditioning".into()))?;

        // The prompt's rows; every other row is zero: the prompt's padding, the condition rows
        // the conditioner writes, and the packing's zero row.
        source.zero_set()?;
        source.slice_set(&conditioning, 0, 0)?;
        text.zero_set()?;
        if self.text_width == self.config.width {
            text.slice_set(&conditioning, 0, 0)?;
        }

        let media = self.media(request, pieces, seed, (pixels, samples))?;
        let cond = Conditioning {
            text,
            source: Some(source),
            media,
            images: None,
            audio: None,
        };

        // The request's own condition draws lead its stream, so its noise is drawn at their end.
        let drawn = Layout {
            condition_elements: self.model.layout(request)?.condition_elements,
            ..*layout
        };
        if !restored {
            self.model
                .draw_noise(seed, &drawn, &video, audio.as_ref())?;
        }
        self.model.prepare_conditioning(request, &cond)?;
        if !restored {
            self.model.prepare_latents(&cond, &video, audio.as_ref())?;
        }
        Ok(())
    }

    /// The raw media of a request's conditions, in the order of `pieces`, each visual piece with
    /// its anchoring draw: each condition's pixels and samples lie after those of the conditions
    /// before it in request order, as the media host lays them out.
    fn media(
        &self,
        request: &VideoRequest,
        pieces: &[ConditionPiece],
        seed: u64,
        (pixels, samples): (Option<Tensor>, Option<Tensor>),
    ) -> Result<Vec<MediaInput>> {
        let mut offsets = vec![(0, 0); request.conditions.len()];
        let mut ordered: Vec<&ConditionPiece> = pieces.iter().collect();
        ordered.sort_by_key(|piece| piece.condition);
        let (mut pixel, mut sample) = (0, 0);
        for piece in ordered {
            let at = offsets.get_mut(piece.condition as usize).ok_or_else(|| {
                Error::Invalid(format!("a piece of condition {}", piece.condition))
            })?;
            *at = (pixel, sample);
            pixel += piece.pixels.as_ref().map_or(0, nn::Shape::elem_count);
            sample += piece.track.as_ref().map_or(0, nn::Shape::elem_count);
        }

        let view = |source: Option<&Tensor>, at: usize, shape: &nn::Shape| -> Result<Tensor> {
            let source = source.ok_or_else(|| {
                Error::Invalid("a condition piece reads media its row names no block of".into())
            })?;
            Ok(source.narrow(0, at, shape.elem_count())?.reshape(shape)?)
        };
        let mut media = Vec::with_capacity(pieces.len());
        for piece in pieces {
            let (pixel, sample) = offsets[piece.condition as usize];
            let noise = match piece.draw {
                Some(draw) => {
                    let dims = [draw.channels, draw.frames, draw.height, draw.width];
                    let out = ops::empty(&dims.map(|d| d as usize)[..], self.config.sample)?;
                    self.model.draw_condition_noise(seed, piece, &out)?;
                    Some(out)
                }
                None => None,
            };
            media.push(MediaInput {
                role: piece.role,
                rows: piece.rows,
                audio_rows: piece.audio_rows,
                pixels: piece
                    .pixels
                    .as_ref()
                    .map(|shape| view(pixels.as_ref(), pixel, shape))
                    .transpose()?,
                audio: piece
                    .track
                    .as_ref()
                    .map(|shape| view(samples.as_ref(), sample, shape))
                    .transpose()?,
                noise,
            });
        }
        Ok(media)
    }

    /// The step `row` names and the request it states, checked against the model's schedule.
    fn staged(&self, bucket: &Bucket, row: &MediaRow) -> Result<(Step, VideoRequest)> {
        let (_, _, warm, _) = self.served(bucket)?;
        let request = stated(warm, &row.conditioning)?;
        let schedule = self.model.schedule(&request)?;

        match schedule.get(row.step as usize) {
            Some(step) if row.steps as usize == schedule.len() => Ok((*step, request)),
            _ => Err(Error::Invalid(format!(
                "request {} asks for step {} of {}, and its schedule has {}",
                row.request.0,
                row.step,
                row.steps,
                schedule.len()
            ))),
        }
    }

    /// What the step of `row` in `bucket` moves around its replay, `last` when it ends the
    /// schedule (module documentation).
    fn moves(
        &self,
        bucket: &Bucket,
        row: &MediaRow,
        tables: &BlockTables,
        last: bool,
    ) -> Result<Moves> {
        let refused = |problem: &str| {
            Error::Invalid(format!(
                "denoising step {} of request {} {problem}",
                row.step, row.request.0
            ))
        };
        let (_, layout, ..) = self.served(bucket)?;
        let [video, audio, text, _] = self.parts(layout)?;
        let (latents, read) = (video.1 + audio.1, text.0 + text.1);
        let slot = address(
            &self
                .memory
                .trajectory(row.slot, 0, &[read as usize], DType::U8)?,
        )?;
        let base = self.state.as_ref().map_or(0, |storage| storage.address());

        let mut before = PitchedBatch::new(SourceOrder::Stream);
        before.push(pitched((base, read), (slot, read), read, 1)?)?;
        let mut after = PitchedBatch::new(SourceOrder::Stream);
        after.push(pitched((slot, latents), (base, latents), latents, 1)?)?;

        // Only the last step names results: the latents it writes.
        let writes = row.results.get(row.reads as usize..).unwrap_or_default();
        if row.reads != 0 || last == writes.is_empty() {
            return Err(refused("names results it does not write"));
        }
        let mut written = Vec::new();
        if last {
            let named = self.named(tables, row, writes, [VIDEO, AUDIO])?;
            for (kind, (offset, bytes)) in [(VIDEO, video), (AUDIO, audio)] {
                // A layout without a track writes nothing into an audio latents block.
                if bytes == 0 {
                    continue;
                }
                let position = named[kind - VIDEO].ok_or_else(|| refused("names no latents"))?;
                let class = self.config.classes[kind];
                let entry = (row.slot, class, position);
                let block =
                    self.memory
                        .lend(tables, entry, true, 0, &[bytes as usize], DType::U8)?;

                after.push(pitched(
                    (address(&block)?, bytes),
                    (base + offset, bytes),
                    bytes,
                    1,
                )?)?;
                written.push((class, position));
            }
        }
        Ok((before, after, written))
    }

    /// The layouts a restore across encodings lays out (`BufferMemory::restore`): the one host
    /// copy, in the part of the host latent region the rank copies with, which every layout that
    /// names the region saves into ([`BufferMemory::part_holder`]), and the rank's own row, both
    /// the request's trajectory as its own layout lays it out (module documentation). The buffer
    /// pool names another encoding only of the class's tensors.
    fn restoring(&self, copy: &BlockCopy) -> Result<(Vec<(u32, Stored)>, Stored)> {
        let refused = |problem: &str| {
            Error::Invalid(format!(
                "request {} restores a trajectory {problem}",
                copy.request.0
            ))
        };
        let ([(host, _)], Some((source, target))) = (copy.runs.as_slice(), copy.encodings) else {
            return Err(refused("from no one block in another encoding"));
        };
        if target != self.encoding || source == self.encoding {
            return Err(refused(
                "between encodings this class does not restore across",
            ));
        }
        let holder = self
            .memory
            .part_holder(host.0)
            .ok_or_else(|| refused("from a host region no part of which the rank copies with"))?;

        let bytes = u64::from(host.2) * self.config.page_bytes;
        let pages = Stored {
            layout: bytes_layout(bytes),
            page_bytes: self.config.page_bytes,
        };
        Ok((vec![(holder, pages.clone())], pages))
    }

    /// Whether `copy` is a `Save`, which reads the slot rows.
    fn saves(&self, copy: &BlockCopy) -> bool {
        copy.runs.iter().any(|(from, _)| self.memory.rows(from.0))
    }

    /// Serves `batch` over `tables` on the stream: returns when the call was issued, and its
    /// output; a batch that names no call carries copies alone.
    fn serve(
        &self,
        batch: &Batch,
        tables: &BlockTables,
    ) -> Result<(Option<Instant>, Vec<CallOutput>)> {
        if batch.requests.is_empty() {
            return Ok((None, Vec::new()));
        }
        let (bucket, row) = self.call(batch)?;
        let stream = self.stream.as_ref().ok_or(Error::State(
            "a denoiser runner with no device denoises nothing",
        ))?;

        if bucket.variant == Variant::LatentPrepare {
            *self.current_opening.borrow_mut() = Some(self.opening(&bucket, row, tables)?);
            let launched = Instant::now();
            let ran = self
                .graphs
                .prepare_inputs(self, &bucket, |_| Ok(()))
                .and_then(|()| self.graphs.eager(self, &bucket));
            self.current_opening.borrow_mut().take();
            ran?;

            let output = CallOutput {
                step_index: row.step,
                ..performed()
            };
            return Ok((Some(launched), vec![output]));
        }

        let (step, request) = self.staged(&bucket, row)?;
        let last = step.index + 1 == row.steps;
        let (before, after, written) = self.moves(&bucket, row, tables, last)?;
        *self.current_step.borrow_mut() = Some((step, request));
        let prepared = self.graphs.prepare_inputs(self, &bucket, |_| Ok(()));
        self.current_step.borrow_mut().take();
        prepared?;

        let launched = Instant::now();
        // SAFETY: every range lies in the slot's row, the state buffer or a block the tables
        // give this call writable (`moves`), which stay mapped until the batch ends; the stream
        // orders the copies before the replay that reads the state buffer and after the one
        // that writes it.
        unsafe { before.submit(stream) }?;
        self.graphs.replay(&bucket)?;
        // SAFETY: as above.
        unsafe { after.submit(stream) }?;
        if self.config.first {
            for (class, position) in written {
                self.memory
                    .share(tables, (row.slot, class, position), Some(stream))?;
            }
        }

        let output = CallOutput {
            step_index: step.index + 1,
            ..performed()
        };
        Ok((Some(launched), vec![output]))
    }
}

impl Runner for DenoiserRunner {
    fn phases(&self) -> &[Phase] {
        &self.config.phases
    }

    fn buckets(&self) -> Vec<Bucket> {
        self.buckets.capture_order().copied().collect()
    }

    /// The latent preparation of each step's layout.
    fn eager_buckets(&self) -> Vec<Bucket> {
        let prepare = |bucket: &Bucket| Bucket {
            variant: Variant::LatentPrepare,
            ..*bucket
        };
        self.buckets.capture_order().map(prepare).collect()
    }

    /// A step's forward advances the latents of the state buffer in place, and a preparation's
    /// opens its request (module documentation); neither returns an output.
    fn forward(&self, bucket: &Bucket) -> Result<Vec<Tensor>> {
        let (_, layout, warm, sets) = self.served(bucket)?;

        if bucket.variant == Variant::Denoise {
            let staged = self.current_step.borrow().clone();
            let (step, request) = match staged {
                Some(staged) => staged,
                None => {
                    let first = self.model.schedule(warm)?.first().copied();
                    let step = first.ok_or_else(|| {
                        Error::Invalid(format!("bucket {bucket}'s schedule has no step"))
                    })?;
                    (step, warm.clone())
                }
            };
            let (video, audio, text, _) = self.state(layout, None)?;
            let cond = Conditioning {
                text,
                source: None,
                media: Vec::new(),
                images: None,
                audio: None,
            };

            self.model
                .forward(step, &request, &video, audio.as_ref(), &cond)?;
            return Ok(Vec::new());
        }

        if let Some((row, request, pieces, products)) = self.current_opening.borrow().clone() {
            let state = self.state(layout, Some(row.slot))?;
            let seed = (row.seed, row.step > 0);
            self.open(layout, (&request, &pieces), seed, state, products)?;
            return Ok(Vec::new());
        }

        // A profile run: the model's initialization, then the candidate and each condition set
        // it may open with, over the state buffer and products of their own extents.
        self.model.initialize()?;
        let requests = std::iter::once(warm.clone()).chain(
            sets.iter()
                .map(|set| opened(warm, set.clone(), warm.text_tokens, Vec::new())),
        );
        for request in requests {
            let pieces = self.model.condition_pieces(&request)?;
            let placeholder = |extent: fn(&ConditionPiece) -> Option<&nn::Shape>| -> Result<_> {
                let total: usize = pieces
                    .iter()
                    .filter_map(extent)
                    .map(nn::Shape::elem_count)
                    .sum();
                match total {
                    0 => Ok(None),
                    total => Ok(Some(ops::empty(total, CONDITION_MEDIA)?)),
                }
            };
            let prompt = (request.text_tokens as usize, self.config.width as usize);
            let products = [
                Some(ops::empty(prompt, self.config.stack)?),
                placeholder(|piece| piece.pixels.as_ref())?,
                placeholder(|piece| piece.track.as_ref())?,
            ];

            let state = self.state(layout, None)?;
            self.open(layout, (&request, &pieces), (0, false), state, products)?;
        }
        Ok(Vec::new())
    }

    /// A step reads no input tensor: its state is the runner's buffer.
    fn idle_inputs(&self, bucket: &Bucket, _columns: &mut [u8]) -> Result<()> {
        self.served(bucket).map(|_| ())
    }

    fn prepare_inputs(&mut self, batch: &Batch, tables: &BlockTables) -> Result<Bucket> {
        let (bucket, row) = self.call(batch)?;
        if bucket.variant == Variant::Denoise {
            let staged = self.staged(&bucket, row)?;
            *self.current_step.borrow_mut() = Some(staged);
        } else {
            *self.current_opening.borrow_mut() = Some(self.opening(&bucket, row, tables)?);
        }

        let prepared = self.graphs.prepare_inputs(&*self, &bucket, |_| Ok(()));
        self.current_step.borrow_mut().take();
        self.current_opening.borrow_mut().take();
        prepared.map(|()| bucket)
    }

    /// One call of `batch` over `tables` (module documentation). It returns once the device
    /// completed the call and the copies [`Runner::copy`] issued for it, whether the call was
    /// served or failed.
    fn execute(&mut self, batch: &Batch, tables: &BlockTables) -> Result<BatchOutput> {
        let started = Instant::now();
        let served = self.serve(batch, tables);

        let ended = match (&self.stream, &self.done) {
            (Some(stream), Some(done)) => done.record(stream).and_then(|()| done.synchronize()),
            _ => Ok(()),
        };
        let (launched, calls) = served?;
        ended?;

        let landed = Instant::now();
        let launched = launched.unwrap_or(landed);
        Ok(BatchOutput {
            calls,
            timing: TimingCounters {
                queued_us: 0,
                device_us: landed.duration_since(launched).as_micros() as u64,
                copy_us: 0,
                host_us: launched.duration_since(started).as_micros() as u64,
            },
            forward_stats: None,
        })
    }

    /// Issues `copies` through the rank's media memory on the runner's stream, ahead of the
    /// batch whose end covers them: a restore of a trajectory saved in another encoding through
    /// the copy manifest, every other copy as its bytes are. A `Save` is the group's first
    /// rank's alone, since it writes the one host copy every rank of the group holds.
    fn copy(&mut self, copies: &[BlockCopy]) -> Result<()> {
        let stream = self.stream.as_ref();
        let plain: Vec<BlockCopy> = copies
            .iter()
            .filter(|copy| copy.encodings.is_none())
            .filter(|copy| self.config.first || !self.saves(copy))
            .cloned()
            .collect();
        let across = copies
            .iter()
            .filter(|copy| copy.encodings.is_some())
            .map(|copy| Ok((copy, self.restoring(copy)?)))
            .collect::<Result<Vec<_>>>()?;

        self.memory.copy(&plain, stream)?;
        for (copy, (sources, target)) in &across {
            self.memory.restore(copy, (&sources[..], target), stream)?;
        }
        Ok(())
    }

    /// The rank's media memory, its trajectory rows included: the denoiser is the media runner
    /// that saves and restores trajectories (`BufferMemory::rows`).
    fn maps(&mut self, region: u8) -> bool {
        self.memory.holds(region) || self.memory.rows(region)
    }

    /// The copies were issued on the runner's stream, which this waits for.
    fn wait_copies(&mut self) -> Result<()> {
        settle(self.stream.as_ref())
    }

    fn graphs(&self) -> &Graphs {
        &self.graphs
    }

    /// The model's tables are computed from its weights, so they are filled again.
    fn weights_changed(&mut self) -> Result<()> {
        self.initialize()
    }
}

/// One bucket per distinct layout of each served request shape (module documentation), with
/// the candidate the first of it states and the condition sets of every candidate it collapses.
fn served(
    model: &dyn Denoiser,
    candidates: &[(VideoRequest, Vec<Vec<ConditionShape>>)],
) -> Result<Vec<Served>> {
    let mut served: Vec<Served> = Vec::new();
    for (request, sets) in candidates {
        let layout = model.layout(request)?;
        let bucket =
            Bucket::denoise(1, layout).serving(request.frames, request.width, request.height);

        match served.iter_mut().find(|(held, ..)| *held == bucket) {
            Some((.., held)) => {
                for set in sets {
                    if !held.contains(set) {
                        held.push(set.clone());
                    }
                }
            }
            None => served.push((bucket, layout, request.clone(), sets.clone())),
        }
    }

    if served.is_empty() {
        return Err(Error::Invalid(
            "a denoiser runner serves no candidate request, so it would capture nothing".into(),
        ));
    }
    Ok(served)
}

/// The bytes of the largest request's state in a slot's row, and the layout of the largest
/// trajectory, among the served buckets.
fn extents<'a>(
    served: &'a [Served],
    config: &DenoiserConfig,
    text_width: u32,
) -> Result<(u64, &'a Layout)> {
    let (mut row, mut largest) = (0, None);
    for (_, layout, ..) in served {
        let [video, audio, _, (offset, source)] = parts(layout, config, text_width)?;
        row = u64::max(row, offset + source);

        let trajectory = video.1 + audio.1;
        if largest.is_none_or(|(held, _)| trajectory > held) {
            largest = Some((trajectory, layout));
        }
    }

    let (_, layout) =
        largest.ok_or_else(|| Error::Invalid("a denoiser runner serves no layout".into()))?;
    Ok((row, layout))
}

/// The parts of a request of `layout` in a slot's row, `(offset, bytes)` each from its first
/// byte (module documentation): the video latents, the audio latents (no byte without a track),
/// the conditioning rows of `text_width` the denoiser reads, and the encoder's rows.
fn parts(layout: &Layout, config: &DenoiserConfig, text_width: u32) -> Result<[(u64, u64); 4]> {
    let (sample, stack) = (element_bytes(config.sample)?, element_bytes(config.stack)?);
    let rows = u64::from(nn::capability::prefix_rows(layout));

    let video = elements(&layout.video) * sample;
    let audio = layout.audio.as_ref().map_or(0, elements) * sample;
    let text = rows * u64::from(text_width) * stack;
    let source = rows * u64::from(config.width) * stack;
    let at = (video + audio).next_multiple_of(UNIT_BYTES);
    let after = (at + text).next_multiple_of(UNIT_BYTES);
    Ok([(0, video), (video, audio), (at, text), (after, source)])
}

/// The request a row's `conditioning` states, in the bucket of `warm`: the conditions, prompt
/// and vision spans the conditioning states, and the resolution, timeline, frame rate and
/// condition rung of the candidate.
fn stated(warm: &VideoRequest, conditioning: &[u8]) -> Result<VideoRequest> {
    let decoded = proto::video::Conditions::decode(conditioning).map_err(|error| {
        Error::Invalid(format!(
            "a video request's conditioning does not decode: {error}"
        ))
    })?;
    let conditions = decoded
        .conditions
        .iter()
        .map(|condition| {
            let (frames, canvas) = condition.media.pixels().unzip();
            ConditionShape {
                role: condition.role,
                frames: frames.unwrap_or(0),
                height: canvas.map_or(0, |canvas| canvas.height),
                width: canvas.map_or(0, |canvas| canvas.width),
                audio_samples: condition
                    .media
                    .audio()
                    .map_or(0, |clip| u32::try_from(clip.samples).unwrap_or(u32::MAX)),
            }
        })
        .collect();

    Ok(opened(
        warm,
        conditions,
        decoded.prompt_tokens,
        decoded.vision_spans,
    ))
}

/// `warm` presenting `conditions`, a prompt of `text_tokens` tokens and `vision_spans`: the role
/// counts follow the conditions, and the rest is the candidate's.
fn opened(
    warm: &VideoRequest,
    conditions: Vec<ConditionShape>,
    text_tokens: u32,
    vision_spans: Vec<(u32, u32)>,
) -> VideoRequest {
    let count = |role: ConditionRole| {
        let counted = conditions.iter().filter(|held| held.role == role).count();
        u32::try_from(counted).unwrap_or(u32::MAX)
    };
    VideoRequest {
        first_frames: count(ConditionRole::FirstFrame),
        last_frames: count(ConditionRole::LastFrame),
        references: count(ConditionRole::Reference),
        text_tokens,
        conditions,
        vision_spans,
        ..warm.clone()
    }
}

/// How a slot's row holds one request's trajectory, as the copy manifest library reads it: the
/// whole video latent, then the whole audio latent when the layout has a track, back to back
/// from the first byte. Every rank of a sequence group holds the whole samples (the velocity is
/// gathered to every rank before the solver's update), so two ranks or epochs whose layouts
/// agree lay out every page alike.
pub fn trajectory_layout(layout: &Layout, dtype: DType) -> Result<StorageLayout> {
    let element = element_bytes(dtype)?;
    let field = |tensor: &str, shape: &LatentShape, offset: u64| {
        let dims = [shape.channels, shape.frames, shape.height, shape.width].map(u64::from);
        let strides = [dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1];
        Field {
            tensor: tensor.to_owned(),
            source: String::new(),
            dtype: ops::abi::name(dtype).to_owned(),
            element,
            offset,
            shape: dims.to_vec(),
            strides: strides.to_vec(),
            logical: Logical {
                shape: dims.to_vec(),
                dims: ["channel", "frame", "row", "column"]
                    .map(str::to_owned)
                    .to_vec(),
                // No parallel configuration divides a sample: every rank holds it whole.
                units: vec![0; 4],
                segments: dims
                    .iter()
                    .map(|extent| {
                        vec![Segment {
                            span: Span::range(0, *extent),
                            local: 0,
                        }]
                    })
                    .collect(),
            },
            encoding: Encoding::Affine,
        }
    };

    let video = elements(&layout.video) * element;
    let mut fields = vec![field("video_latents", &layout.video, 0)];
    let mut bytes = video;
    if let Some(audio) = &layout.audio {
        fields.push(field("audio_latents", audio, video));
        bytes += elements(audio) * element;
    }
    Ok(StorageLayout { bytes, fields })
}

/// `bytes` bytes of a trajectory, whole on the rank that holds them: what each rank's part of a
/// saved block holds, whatever the group that saved it.
fn bytes_layout(bytes: u64) -> StorageLayout {
    let field = Field {
        tensor: "trajectory".to_owned(),
        source: String::new(),
        dtype: ops::abi::name(DType::U8).to_owned(),
        element: 1,
        offset: 0,
        shape: vec![bytes],
        strides: vec![1],
        logical: Logical {
            shape: vec![bytes],
            dims: vec!["byte".to_owned()],
            units: vec![0],
            segments: vec![vec![Segment {
                span: Span::range(0, bytes),
                local: 0,
            }]],
        },
        encoding: Encoding::Affine,
    };
    StorageLayout {
        bytes,
        fields: vec![field],
    }
}

#[cfg(test)]
mod tests;

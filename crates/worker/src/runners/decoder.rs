//! `DecoderRunner`: the runner of every decoder, a video decoder with its splicer and an audio
//! decoder alike.
//!
//! The runner holds its model through `nn::Decoder`, and a video decoder's splicer through
//! `nn::Splicer`; it names no concrete model. A media row names the products its call reads and
//! writes as positions in the request's product table (`ipc::batch::MediaRow::results`), and the
//! runner finds their blocks through the rank's device block tables, in the product buffer of
//! the rank's [`BufferMemory`].
//!
//! ## A call
//!
//! A video decode is a round of the timeline's units: `row.unit` and the units after it, one per
//! splice tail the row writes. An audio decode is the request's track, one unit. A row that names
//! no rank is run by every rank of the decoder group, and each decodes its share of the round
//! ([`RoundConfig`]); a row that names a rank is run by that rank alone, which decodes every unit
//! of it. Each unit a rank decodes is issued on the runner's stream, in unit order:
//!
//! 1. **The decode.** The unit's latents are copied into the runner's latent buffer (a clip's
//!    window of the trajectory, the latent frames from `Decoder::clip_start`, or the whole track)
//!    and its bucket's graph replays: the window gathered and decoded into the runner's raster
//!    buffer, or the track decoded and interleaved into 16-bit PCM. A clip that a successor fades
//!    against then copies the raster's last frames into its tail block.
//! 2. **The splice** of a clip a splicer places. A clip whose splice fades against its
//!    predecessor has the predecessor's tail copied into the runner's tail buffer, and the splice
//!    runs eagerly: the raster placed under the unit's splice and packed into bytes.
//! 3. The unit's output is copied into its product: the bytes of its splice into the video units
//!    block from the splice's first frame, the raster of a clip no splicer places, or the PCM into
//!    the audio samples block.
//!
//! Every address is computed and every block checked before anything is issued: a position the
//! tables hold no entry for, a write to a read-only entry, or bytes past a block refuse the
//! batch. The copies are the driver's, so the forwards' launches are the only kernels of a call.
//!
//! ## Why a clip's splice runs eagerly
//!
//! A splice's runs, their number and their frames, follow from the clip's place in its timeline
//! (`Splicer::splice`), which is a request's value, and a captured graph's launches follow from
//! its bucket alone: a graph would place the clip under every distinct splice of the deployment's
//! timelines to keep one. The splice runs eagerly instead, as the forward of a splice bucket
//! ([`Runner::eager_buckets`]): a clip's bucket stating the timeline it serves, whose forward
//! places the splice `serve` stages, a few launches beside a clip's decode. Its profile places
//! every distinct splice of the timeline, so the runner's region holds the largest and every
//! generation resolves their kernels. The raster lies in the runner's buffers, outside the
//! region a forward allocates from, so the splice reads it where the decode left it.
//!
//! ## What the products hold
//!
//! | Product | Bytes |
//! | --- | --- |
//! | Video latents | The trajectory, `[channels, frames · rows · columns]` in `Decoder::latent_type`, `frames` the timeline's latent frames ([`DecoderConfig::timelines`]) |
//! | Video units | The timeline's frames, `[frames, rows, columns, 3]` bytes, each unit's kept frames from its splice's `first`; without a splicer, unit `u`'s raster in the decoder's element type at `u` times its bytes |
//! | Video tail of unit `u` | The last frames of unit `u`'s raster that unit `u + 1` fades against, `[3, frames, rows, columns]` in the decoder's element type |
//! | Audio latents | The track's elements in `Decoder::latent_type` |
//! | Audio samples | `[samples, 2]` interleaved `i16` |
//!
//! ## The tail a unit fades against
//!
//! Each rank writes the tail of a unit it decodes into its own run of the unit's tail block. The
//! unit after it reads that tail in one of three ways:
//!
//! - **The same rank, the same call.** The unit after it on the rank's stream reads the run the
//!   stream wrote.
//! - **Another rank, the same call.** A round's units are dealt to the ranks in order, so the
//!   first unit of a rank's share follows the last of the share before it. The rank that decodes
//!   the predecessor copies its tail into the successor's rank's run of the tail block behind its
//!   own tail copy, which follows the decode alone, and hands that rank an event recorded behind
//!   it ([`BufferMemory::hand_off`]) before it waits for any tail itself. The successor's rank
//!   decodes its unit first, then waits on the host for the hand-off, which is the other rank's
//!   issue and not its completion, and its stream waits for the event before the splice's copies
//!   ([`BufferMemory::take_hand_off`]). This is the only wait between executors
//!   (`specs/engine-worker-port.md` §2.2): the ranks of a round decode at once, and a splice
//!   waits for its predecessor's decode, never for another splice.
//! - **An earlier call.** The first unit of a round reads the tail of the round before it, which
//!   the scheduler gives it only once that round completed, and `execute` returns only once its
//!   stream passed every copy: the write is ordered before the read by completion, and the read
//!   finds the tail where it lies, in the run of the rank that decoded the round's last unit,
//!   which the device reaches by peer access. A row that names a rank reads it from that rank's
//!   own run, where its earlier rounds wrote it.

use std::cell::Cell;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ipc::batch::{Batch, BatchOutput, Extent, MediaRow, Phase, TimingCounters};
use nn::{Clip, Decoder, Layout, Splice, Splicer};
use ops::pcm::{self, PcmInterleave};
use ops::rgb::{self, RgbBytes};
use platform::{Event, EventFlags, Pitched, PitchedBatch, SourceOrder, Stream};
use uniserve_tensor::{DType, Device, Shape, Tensor};

use crate::buckets::{Bucket, Buckets};
use crate::graphs::Graphs;
use crate::memory::allocation::Storage;
use crate::memory::buffers::{BufferMemory, UNIT_BYTES};
use crate::memory::lend;
use crate::memory::tables::{BlockCopy, BlockTables};
use crate::runner::Runner;
use crate::runners::{address, element_bytes, performed};
use crate::{Error, Result};

/// The channels of a decoded raster and of its packed bytes: red, green and blue.
const CHANNELS: u64 = rgb::CHANNELS as u64;

/// The channels of the interleaved PCM.
const PCM_CHANNELS: u32 = 2;

/// How long a unit's rank waits for the tail another rank hands it before it names the rank that
/// never handed one: the other rank's issue of one round's units, behind whatever batch it still
/// runs.
const TAIL_WAIT: Duration = Duration::from_secs(120);

/// What one unit of a call issues, in the order `serve` issues it (module documentation):
///
/// - the copies before its decode: the clip's window or the track into the latent buffer;
/// - the tail it leaves its successor: the copy of the raster's last frames into its tail block
///   behind the decode, and, when another rank decodes the successor in this call, the tail's
///   address, the rank it is handed to, the tail block's position and its bytes;
/// - its splice: the position of the tail block whose hand-off it takes first, the copy of the
///   predecessor's tail into the tail buffer, and the splice it places, `None` for a unit whose
///   decode is its output;
/// - the address in the product buffer its output lands at.
type Unit = (
    PitchedBatch,
    (PitchedBatch, Option<(u64, u32, u32, u64)>),
    (Option<u32>, PitchedBatch, Option<Splice>),
    u64,
);

/// What a deployment fixes of a decoder runner. Immutable once the runner is built.
#[derive(Clone, Debug)]
pub struct DecoderConfig {
    /// The phases the runner serves, as the scheduler names them in a batch.
    pub phases: Vec<Phase>,
    /// The buckets the runner captures: one clip of the decoder's window per canvas
    /// (`Bucket::clip`), or one per track (`Bucket::audio`).
    pub buckets: Buckets,
    /// The classes of the products a call reads and writes: the latents it decodes, the media it
    /// writes (the video units or the audio samples), and a video's splice tails.
    pub classes: [u8; 3],
    /// For a video decoder, each timeline the deployment serves, in frames, with the latent
    /// frames of its trajectory as the denoiser lays the request out: the splices of its units,
    /// and the stride of the trajectory in its latents block.
    pub timelines: Vec<(u32, u32)>,
    /// For a video decoder, how a round whose row names no rank is dealt to the decoder group's
    /// ranks. The runner decodes the share of the rank whose media memory it holds, which is one
    /// of the round's ranks.
    pub round: RoundConfig,
}

/// How the units of a video's decode rounds are dealt to the ranks of the decoder group: the
/// deployment's video decoder block, its ranks in order and the units each decodes in a round.
/// The scheduler cuts a timeline into rounds of [`RoundConfig::width`] units from unit 0; the
/// decoders deal each round's units in order, and the media host finds each unit it encodes in
/// the run of the rank that decoded it. Immutable once a runner is built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoundConfig {
    /// The ranks a round's units are dealt to, in order, each by its rank index in the process's
    /// media memory (`BufferMemory::rank`).
    pub ranks: Vec<u32>,
    /// The consecutive units each rank decodes in a round.
    pub units: u32,
}

impl RoundConfig {
    /// The units of one round: `units` for each of `ranks`.
    pub fn width(&self) -> u32 {
        let ranks = u32::try_from(self.ranks.len()).unwrap_or(u32::MAX);

        ranks.saturating_mul(self.units)
    }

    /// The rank that decodes unit `unit` of a timeline: the rank of `ranks` whose share of its
    /// round holds it. `None` for a deal of no unit.
    pub fn rank_of(&self, unit: u32) -> Option<u32> {
        let position = unit.checked_rem(self.width())?.checked_div(self.units)?;

        self.ranks.get(position as usize).copied()
    }

    /// Whether the configuration deals a round: at least one unit to each rank, and each rank
    /// once.
    pub(crate) fn deals(&self) -> bool {
        let once = (0..self.ranks.len()).all(|at| !self.ranks[..at].contains(&self.ranks[at]));

        self.units > 0 && !self.ranks.is_empty() && once
    }
}

/// The runner of a decoder (module documentation).
pub struct DecoderRunner {
    model: Arc<dyn Decoder + Send + Sync>,
    /// A video decoder's splicer; `None` for a track's decoder, or a clip's that places none.
    splicer: Option<Arc<dyn Splicer + Send + Sync>>,
    config: DecoderConfig,
    graphs: Graphs,
    /// The rank's media memory, which every media runner of the rank shares.
    memory: Arc<BufferMemory>,
    /// The stream the graphs replay on; `None` for graphs that hold no device.
    stream: Option<Stream>,
    /// Recorded behind a batch's last copy: what `execute` waits for.
    done: Option<Event>,
    /// Each bucket's latent layout: the model's map of a clip, or a track's own.
    layouts: Vec<(Bucket, Layout)>,
    /// The splice a splice bucket's forward places, from frame zero, while `serve` runs it;
    /// `None` outside, when the forward places every distinct splice of its timeline.
    staged: Cell<Option<Splice>>,
    /// The latent buffer from byte 0, the tail buffer from byte `tail` and the raster buffer
    /// from byte `raster`, `bytes` in all; `None` for graphs that hold no device.
    buffers: Option<Arc<Storage>>,
    tail: u64,
    raster: u64,
    bytes: u64,
}

impl DecoderRunner {
    /// A runner of `model`, with `splicer` for a video decoder that places its clips, under
    /// `config`, over `memory`.
    ///
    /// - `graphs` are made for no input tensors: a decode's host values are its forward's.
    /// - `stream` is the stream `graphs` replay on, on which the runner issues its copies; `None`
    ///   for graphs that hold no device, whose runner computes over meta tensors.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for no bucket, a bucket that is neither a clip nor a track, a clip's
    /// that states a request (a splice's, [`Runner::eager_buckets`]), a video
    /// decoder serving no timeline or with a round that deals no unit to its memory's rank, a
    /// splicer of another element type, a track that is not stereo, a stream without a device or
    /// the reverse; what the model or the splicer refuses:
    /// a canvas the model lays out no clip of, a timeline the splicer has no splices of; the
    /// driver's refusal of the buffers.
    pub fn new(
        model: Arc<dyn Decoder + Send + Sync>,
        splicer: Option<Arc<dyn Splicer + Send + Sync>>,
        config: DecoderConfig,
        graphs: Graphs,
        stream: Option<Stream>,
        memory: Arc<BufferMemory>,
    ) -> Result<Self> {
        if graphs.enter().is_ok() != stream.is_some() || config.buckets.is_empty() {
            return Err(Error::Invalid(
                "a decoder runner takes buckets, and the stream of its graphs or none for graphs \
                 that hold no device"
                    .into(),
            ));
        }
        if let Some(splicer) = &splicer
            && splicer.element_type() != model.element_type()
        {
            return Err(Error::Invalid(
                "the splicer places rasters of another element type than the decoder's".into(),
            ));
        }

        // The most frames a splice of the deployment's timelines fades.
        let mut overlap = 0;
        if let Some(splicer) = &splicer {
            for &(frames, _) in &config.timelines {
                let splices = splicer.splices(frames)?;
                let faded = splices.iter().map(|splice| splice.overlap).max();
                overlap = overlap.max(faded.unwrap_or(0));
            }
        }

        // Each bucket's layout, and the bytes its latents, a clip's raster and the tail a clip
        // fades against take.
        let latent = element_bytes(model.latent_type())?;
        let element = element_bytes(model.element_type())?;
        let (mut layouts, mut latents, mut tails, mut rasters) = (Vec::new(), 0, 0, 0);
        for bucket in config.buckets.iter().copied() {
            let layout = match bucket.extent {
                Extent::Clip {
                    frames,
                    height,
                    width,
                } if config.timelines.is_empty() => {
                    return Err(Error::Invalid(format!(
                        "a video decoder of {frames}x{height}x{width} clips serves no timeline"
                    )));
                }
                // A clip's bucket that states a request is a splice's (`Runner::eager_buckets`).
                Extent::Clip { .. } if bucket.serves.is_some() => {
                    return Err(Error::Invalid(format!(
                        "bucket {bucket} states the request a splice serves, and no clip's decode"
                    )));
                }
                Extent::Clip { .. }
                    if !config.round.deals() || !config.round.ranks.contains(&memory.rank()) =>
                {
                    return Err(Error::Invalid(format!(
                        "a video decoder of rank {} deals its rounds as {:?}, which is no deal of \
                         at least one unit to each of distinct ranks with this rank among them",
                        memory.rank(),
                        config.round
                    )));
                }
                Extent::Clip {
                    frames,
                    height,
                    width,
                } => {
                    let layout = model.layout_of(Clip {
                        frames,
                        height,
                        width,
                    })?;
                    let plane = CHANNELS * u64::from(height) * u64::from(width) * element;
                    let video = layout.video;
                    latents = latents.max(u64::from(video.channels) * voxels(&video) * latent);
                    tails = tails.max(u64::from(overlap) * plane);
                    rasters = rasters.max(u64::from(frames) * plane);
                    layout
                }
                Extent::Latents(layout) => {
                    let track = layout.audio.ok_or_else(|| {
                        Error::Invalid(format!("bucket {bucket} states no audio track to decode"))
                    })?;
                    if model.pcm_window(&track)?.1 != PCM_CHANNELS {
                        return Err(Error::Invalid(format!(
                            "bucket {bucket}'s track is not the stereo track PCM interleaves"
                        )));
                    }
                    latents = latents.max(elements(&track) * latent);
                    layout
                }
                _ => {
                    return Err(Error::Invalid(format!(
                        "bucket {bucket} is no decode of a clip or of a track"
                    )));
                }
            };
            layouts.push((bucket, layout));
        }
        let tail = latents.next_multiple_of(UNIT_BYTES);
        let raster = (tail + tails).next_multiple_of(UNIT_BYTES);
        let bytes = raster + rasters;

        // The buffers hold defined values for the capture's first forward.
        let buffers = match &stream {
            Some(stream) => {
                let length = usize::try_from(bytes)
                    .map_err(|_| Error::Invalid("the decoder's buffers exceed memory".into()))?;
                let storage = Storage::linear(stream.device(), length)?;

                // SAFETY: the range is the whole allocation, which this runner keeps for as long
                // as the graphs captured over it exist.
                unsafe { platform::memset_d8(storage.address(), 0, length, stream) }?;
                Some(storage)
            }
            None => None,
        };

        Ok(Self {
            done: stream.as_ref().map(|_| Event::new(EventFlags::default())),
            model,
            splicer,
            config,
            graphs,
            memory,
            stream,
            layouts,
            staged: Cell::new(None),
            buffers,
            tail,
            raster,
            bytes,
        })
    }

    /// The bytes of device memory the runner holds beside its graphs' region: its latent, tail
    /// and raster buffers, sized for its largest bucket.
    pub const fn device_bytes(&self) -> u64 {
        self.bytes
    }

    /// The latent layout of `bucket`.
    fn layout(&self, bucket: &Bucket) -> Result<Layout> {
        self.layouts
            .iter()
            .find(|(held, _)| held == bucket)
            .map(|(_, layout)| *layout)
            .ok_or_else(|| Error::Invariant(format!("bucket {bucket} is none the runner states")))
    }

    /// The tensor of `shape` elements of `dtype` from byte `offset` of the runner's buffers.
    fn buffer(&self, offset: u64, shape: impl Into<Shape>, dtype: DType) -> Result<Tensor> {
        let shape = shape.into();
        if offset + shape.elem_count() as u64 * element_bytes(dtype)? > self.bytes {
            return Err(Error::Invariant(format!(
                "{shape:?} {dtype:?} from byte {offset} lies past the decoder's buffers"
            )));
        }
        let base = self.buffers.as_ref().map_or(0, |storage| storage.address());

        // SAFETY: the tensor lies inside the buffers, which this runner owns for as long as its
        // graphs exist; only the copies it issues on its stream and its graphs write them.
        unsafe {
            lend(
                base + offset,
                shape,
                dtype,
                self.stream.as_ref().map(Stream::device),
            )
        }
    }

    /// The address of `bytes` bytes from byte `offset` of the block `row`'s slot holds at
    /// `position` of `class`, which the call writes when `writes`.
    fn place(
        &self,
        tables: &BlockTables,
        row: &MediaRow,
        (class, position): (u8, u32),
        writes: bool,
        (offset, bytes): (u64, u64),
    ) -> Result<u64> {
        let length = usize::try_from(bytes)
            .map_err(|_| Error::Invalid("a product's bytes exceed memory".into()))?;
        let block = self.memory.lend(
            tables,
            (row.slot, class, position),
            writes,
            offset,
            &[length],
            DType::U8,
        )?;
        address(&block)
    }

    /// The address of the first `bytes` bytes of the run on rank `rank` of the process of the
    /// block `row`'s slot holds at `position` of `class`, which the call reads.
    fn read_on(
        &self,
        tables: &BlockTables,
        row: &MediaRow,
        (class, position): (u8, u32),
        rank: u32,
        bytes: u64,
    ) -> Result<u64> {
        let runs = tables.runs(row.slot, class, position);
        let run = self.memory.run_on(runs, rank).ok_or_else(|| {
            Error::Invalid(format!(
                "the block of slot {} at position {position} of class {class} has no run on rank \
                 {rank} of this process",
                row.slot
            ))
        })?;

        self.memory.address(run, 0, bytes)
    }

    /// Checks `batch` and states its call: the bucket that holds it and the units it decodes.
    fn units(&self, batch: &Batch, tables: &BlockTables) -> Result<(Bucket, Vec<Unit>)> {
        batch.check()?;
        let bucket = self.config.buckets.select(&batch.shape).ok_or_else(|| {
            Error::Invalid(format!("no bucket of the decoder holds {}", batch.shape))
        })?;
        let [row] = batch.media.as_slice() else {
            return Err(Error::Invalid(format!(
                "batch {} decodes {} requests in a bucket of one",
                batch.id.0,
                batch.media.len()
            )));
        };

        let decoded = match bucket.extent {
            Extent::Clip { .. } => self.round(&bucket, row, tables)?,
            _ => vec![self.track(&bucket, row, tables)?],
        };
        Ok((bucket, decoded))
    }

    /// The units of a video decode round (module documentation).
    fn round(&self, bucket: &Bucket, row: &MediaRow, tables: &BlockTables) -> Result<Vec<Unit>> {
        let refused = |problem: String| Error::Invalid(format!("a video decode {problem}"));
        let Extent::Clip {
            frames,
            height,
            width,
        } = bucket.extent
        else {
            return Err(Error::Invariant(format!("bucket {bucket} is no clip")));
        };
        let [latents_class, media_class, tail_class] = self.config.classes;

        // The results: the latents and, after the first round, the tail before the round; then
        // the video units and one tail per unit the round decodes.
        let reads = row.reads as usize;
        let (Some(&latents), Some((&media, tails))) = (
            row.results.first(),
            row.results.get(reads..).and_then(<[u32]>::split_first),
        ) else {
            return Err(refused(format!("row names {:?}", row.results)));
        };
        if tails.is_empty() || reads != 1 + usize::from(row.unit > 0) {
            return Err(refused(format!(
                "of unit {} reads {reads} results and writes {} tails",
                row.unit,
                tails.len()
            )));
        }
        let trajectory = self
            .config
            .timelines
            .iter()
            .find(|(served, _)| *served == row.frames)
            .map(|(_, latent)| *latent)
            .ok_or_else(|| refused(format!("of {} frames serves no timeline", row.frames)))?;
        let schedule = match &self.splicer {
            Some(splicer) => Some(splicer.splices(row.frames)?),
            None => None,
        };

        // The rank that decodes each unit: for a row that names a rank, that rank, which the
        // executor hands the batch to alone; otherwise the round's deal, whose rounds start
        // every `width` units, so a row's round covers no more and starts at a round's first.
        let own = self.memory.rank();
        let round = &self.config.round;
        let dealt = round.width();
        let starts = row.unit.checked_rem(dealt) == Some(0);
        if row.rank.is_none() && (!starts || tails.len() > dealt as usize) {
            return Err(refused(format!(
                "of {} units from unit {} is no round of {dealt} units",
                tails.len(),
                row.unit
            )));
        }
        let decoder = |unit: u32| match row.rank {
            Some(_) => Some(own),
            None => round.rank_of(unit),
        };

        // A latent frame's bytes of one channel, and a decoded frame's in the raster and packed.
        let video = self.layout(bucket)?.video;
        let plane = u64::from(video.height) * u64::from(video.width);
        let plane = plane * element_bytes(self.model.latent_type())?;
        let (window, pitch) = (
            u64::from(video.frames) * plane,
            u64::from(trajectory) * plane,
        );
        let pixels = u64::from(height) * u64::from(width);
        let raster = pixels * element_bytes(self.model.element_type())?;
        let base = self.buffers.as_ref().map_or(0, |storage| storage.address());
        let copies = || PitchedBatch::new(SourceOrder::Stream);

        let mut units = Vec::with_capacity(tails.len());
        for (index, unit) in (row.unit..).take(tails.len()).enumerate() {
            if decoder(unit) != Some(own) {
                continue;
            }
            let mut before = copies();

            // The clip's window: its latent frames of every channel, a trajectory's row apart.
            let first = self.model.clip_start(unit);
            if first + video.frames > trajectory {
                return Err(refused(format!(
                    "of unit {unit} reads latent frames past the trajectory's {trajectory}"
                )));
            }
            let extent = u64::from(video.channels).saturating_sub(1) * pitch + window;
            let source = self.place(
                tables,
                row,
                (latents_class, latents),
                false,
                (u64::from(first) * plane, extent),
            )?;
            before.push(pitched(
                (base, window),
                (source, pitch),
                window,
                video.channels,
            )?)?;

            let Some(schedule) = &schedule else {
                // A clip no splicer places is kept as its raster, unit after unit.
                let bytes = CHANNELS * u64::from(frames) * raster;
                let at = (u64::from(unit) * bytes, bytes);
                let kept = self.place(tables, row, (media_class, media), true, at)?;
                units.push((before, (copies(), None), (None, copies(), None), kept));
                continue;
            };
            let splice = schedule.get(unit as usize).copied().ok_or_else(|| {
                refused(format!(
                    "of unit {unit} lies past its {} units",
                    schedule.len()
                ))
            })?;
            let successor = schedule
                .get(unit as usize + 1)
                .map_or(0, |next| next.overlap);
            if successor > frames {
                return Err(refused(format!(
                    "of unit {unit} leaves a tail past its clip"
                )));
            }

            // The tail it fades against: the round's own before it, or the one before the round,
            // found as the module documentation states by the rank that decodes its predecessor.
            let (mut taken, mut fade) = (None, copies());
            if splice.overlap > 0 {
                let previous = match index {
                    0 => row.results.get(1),
                    _ => tails.get(index - 1),
                };
                let (Some(&previous), Some(before_rank)) =
                    (previous, unit.checked_sub(1).and_then(decoder))
                else {
                    return Err(refused(format!(
                        "of unit {unit} fades and names no tail before it"
                    )));
                };
                let bytes = CHANNELS * u64::from(splice.overlap) * raster;
                let read = (tail_class, previous);
                let source = match before_rank {
                    rank if rank == own => self.place(tables, row, read, false, (0, bytes))?,
                    _ if index > 0 => {
                        taken = Some(previous);
                        self.read_on(tables, row, read, own, bytes)?
                    }
                    rank => self.read_on(tables, row, read, rank, bytes)?,
                };
                fade.push(pitched(
                    (base + self.tail, bytes),
                    (source, bytes),
                    bytes,
                    1,
                )?)?;
            }

            // The bytes of its own splice, from the splice's first frame of the timeline.
            let at = (
                u64::from(splice.first) * pixels * CHANNELS,
                u64::from(splice.keep) * pixels * CHANNELS,
            );
            let kept = self.place(tables, row, (media_class, media), true, at)?;

            // The tail it leaves its successor: the end of each channel's frames of the raster,
            // `[3, frames, rows, columns]`, handed to the successor's rank when another rank
            // decodes it in this call; one the next round reads stays where it lies.
            let (mut left, mut handed) = (copies(), None);
            if successor > 0 {
                // A channel's last frames and its whole frames, then the tail's bytes.
                let (end, length) = (u64::from(successor) * raster, u64::from(frames) * raster);
                let (position, bytes) = (tails[index], CHANNELS * end);
                let at = self.place(tables, row, (tail_class, position), true, (0, bytes))?;
                left.push(pitched(
                    (at, end),
                    (base + self.raster + length - end, length),
                    end,
                    CHANNELS as u32,
                )?)?;

                let reader = decoder(unit + 1).filter(|&rank| rank != own);
                if let Some(reader) = reader.filter(|_| index + 1 < tails.len()) {
                    // The reader backs the tail block: checked before anything is issued.
                    self.read_on(tables, row, (tail_class, position), reader, bytes)?;
                    handed = Some((at, reader, position, bytes));
                }
            }
            let splice = Some(Splice { first: 0, ..splice });
            units.push((before, (left, handed), (taken, fade, splice), kept));
        }
        Ok(units)
    }

    /// The one unit of an audio decode: the track's latents, and its PCM.
    fn track(&self, bucket: &Bucket, row: &MediaRow, tables: &BlockTables) -> Result<Unit> {
        let track = self
            .layout(bucket)?
            .audio
            .ok_or_else(|| Error::Invariant(format!("bucket {bucket} states no track")))?;
        let [latents_class, media_class, _] = self.config.classes;
        let (Some(&latents), Some(&samples), 1) =
            (row.results.first(), row.results.get(1), row.reads)
        else {
            return Err(Error::Invalid(format!(
                "an audio decode reads its latents and writes its samples, and its row names {:?}",
                row.results
            )));
        };

        let bytes = elements(&track) * element_bytes(self.model.latent_type())?;
        let source = self.place(tables, row, (latents_class, latents), false, (0, bytes))?;
        let (window, channels) = self.model.pcm_window(&track)?;
        let pcm = (0, window * u64::from(channels) * 2);
        let kept = self.place(tables, row, (media_class, samples), true, pcm)?;

        let base = self.buffers.as_ref().map_or(0, |storage| storage.address());
        let copies = || PitchedBatch::new(SourceOrder::Stream);
        let mut before = copies();
        before.push(pitched((base, bytes), (source, bytes), bytes, 1)?)?;
        Ok((before, (copies(), None), (None, copies(), None), kept))
    }

    /// Waits for the tail another rank hands this one into its run of the tail block `slot`
    /// holds at `taken`, when the unit takes one, and orders the runner's stream after its copy.
    fn take(&self, tables: &BlockTables, slot: u32, taken: Option<u32>) -> Result<()> {
        let Some(position) = taken else {
            return Ok(());
        };
        let entry = (slot, self.config.classes[2], position);

        self.memory
            .take_hand_off(tables, entry, self.stream.as_ref(), TAIL_WAIT)
    }

    /// Hands the tail a unit left in this rank's run at the address `handed` names to the rank it
    /// names, which decodes the unit's successor in the same call: copied into that rank's run of
    /// the tail block on the runner's stream, behind the unit's own tail copy.
    fn hand(
        &self,
        tables: &BlockTables,
        slot: u32,
        handed: Option<(u64, u32, u32, u64)>,
    ) -> Result<()> {
        let Some((at, reader, position, bytes)) = handed else {
            return Ok(());
        };
        let entry = (slot, self.config.classes[2], position);

        // SAFETY: `at` is the first byte of this rank's run of the tail block, which the unit's
        // copies wrote on this stream before and nothing writes again until the batch ended: the
        // block is the request's, and its next write is a later call's.
        unsafe {
            self.memory
                .hand_off(tables, entry, (at, bytes), reader, self.stream.as_ref())
        }?;
        Ok(())
    }

    /// Serves `batch`: every unit's decode, tail and splice on the stream, with the hand-offs of
    /// the tails between the ranks (module documentation). Returns when the units were issued,
    /// for a batch that names a call. A runner whose graphs hold no device issues no copy, and
    /// its graphs refuse the first replay.
    fn serve(&self, batch: &Batch, tables: &BlockTables) -> Result<Option<Instant>> {
        // A batch that names no call carries copies alone.
        if batch.requests.is_empty() {
            return Ok(None);
        }
        let (bucket, units) = self.units(batch, tables)?;
        let (slot, frames) = batch
            .media
            .first()
            .map_or((0, 0), |row| (row.slot, row.frames));

        // Every copy below is safe for one reason: each range lies in a block the tables give
        // this call or in the runner's buffers (`units`), which stay mapped until the batch
        // ends, and a forward's outputs stay until the next forward. The stream orders each
        // copy after the forward before it, and a buffer's next write after the forward that
        // read it.
        let launched = Instant::now();
        for (before, (left, handed), (taken, fade, splice), kept) in &units {
            // The decode reads the unit's latents alone, so it waits for no other rank; the tail
            // it leaves is handed on before this rank waits for one.
            self.graphs.prepare_inputs(self, &bucket, |_| Ok(()))?;
            // SAFETY: as stated above the loop.
            unsafe { self.submit(before) }?;
            let mut outputs = self.graphs.replay(&bucket)?;

            // SAFETY: as stated above the loop.
            unsafe { self.submit(left) }?;
            self.hand(tables, slot, *handed)?;

            // The splice fades against its predecessor's tail, behind that tail's hand-off.
            if let Some(splice) = *splice {
                self.take(tables, slot, *taken)?;
                // SAFETY: as stated above the loop.
                unsafe { self.submit(fade) }?;
                outputs = self.splice(&bucket, frames, splice)?;
            }

            // SAFETY: as stated above the loop.
            unsafe { self.submit(&after(&outputs, *kept)?) }?;
        }
        Ok(Some(launched))
    }

    /// Submits `copies` on the runner's stream; with graphs that hold no device, none.
    ///
    /// # Safety
    ///
    /// As [`PitchedBatch::submit`]: every range stays valid until the stream passed the copies.
    unsafe fn submit(&self, copies: &PitchedBatch) -> Result<()> {
        let Some(stream) = &self.stream else {
            return Ok(());
        };

        // SAFETY: forwarded from the caller.
        Ok(unsafe { copies.submit(stream) }?)
    }

    /// Places `splice` of the clip `clip`'s graph decoded last into a timeline of `frames`
    /// frames: the eager forward of the splice bucket with `splice` staged. Returns its bytes.
    fn splice(&self, clip: &Bucket, frames: u32, splice: Splice) -> Result<Vec<Tensor>> {
        let bucket = splicing(clip, frames)
            .ok_or_else(|| Error::Invariant(format!("bucket {clip} splices no clip")))?;

        self.staged.set(Some(splice));
        let placed = self
            .graphs
            .prepare_inputs(self, &bucket, |_| Ok(()))
            .and_then(|()| self.graphs.eager(self, &bucket));
        self.staged.set(None);
        placed
    }

    /// The clip forward of `layout` decoding into the raster buffer, `[3, frames, rows, columns]`
    /// of `extent`, `(frames, rows, columns)`; it returns the raster.
    fn decode_clip(
        &self,
        layout: &Layout,
        (frames, height, width): (u32, u32, u32),
    ) -> Result<Vec<Tensor>> {
        let video = layout.video;
        let (latent, dtype) = (self.model.latent_type(), self.model.element_type());

        // The window the unit's copy left in the latent buffer, gathered into the decode's rows.
        let channels = video.channels as usize;
        let window = self.buffer(0, (channels, voxels(&video) as usize), latent)?;
        let rows = match self.model.latent_rows(layout)? {
            Some(rows) => rows.elem_count(),
            None => voxels(&video) as usize,
        };
        let block = backend::empty((rows, channels), latent)?;
        self.model.gather_latents(layout, &window, 0, &block)?;

        let shape = (
            CHANNELS as usize,
            frames as usize,
            height as usize,
            width as usize,
        );
        let raster = self.buffer(self.raster, shape, dtype)?;
        self.model.decode(layout, &block, Some(&raster), None)?;
        Ok(vec![raster])
    }

    /// The splice forward of a clip of `extent`, `(frames, rows, columns)`, into a timeline of
    /// `timeline` frames: the staged splice, or every distinct splice of the timeline, placed
    /// over the raster buffer and the tail buffer; it returns each splice's bytes,
    /// `[keep, rows, columns, 3]`.
    fn splice_clip(
        &self,
        timeline: u32,
        (frames, height, width): (u32, u32, u32),
    ) -> Result<Vec<Tensor>> {
        let splicer = self.splicer.as_ref().ok_or_else(|| {
            Error::Invariant("a decoder that places no clip runs no splice".into())
        })?;
        let splices = match self.staged.get() {
            Some(splice) => vec![splice],
            None => {
                let mut distinct = Vec::new();
                for splice in splicer.splices(timeline)? {
                    let splice = Splice { first: 0, ..splice };
                    if !distinct.contains(&splice) {
                        distinct.push(splice);
                    }
                }
                distinct
            }
        };

        let (rgb, height, width) = (CHANNELS as usize, height as usize, width as usize);
        let dtype = self.model.element_type();
        let raster = self.buffer(self.raster, (rgb, frames as usize, height, width), dtype)?;

        // The de-normalization's constants: the three means, then the three deviations.
        let norm = splicer.pixel_norm();
        let constants: Vec<f32> = norm.mean.iter().chain(&norm.std).copied().collect();
        let constants =
            Tensor::from_slice(&constants, 2 * rgb, &Device::Cpu)?.to_device(raster.device())?;
        let (mean, std) = (constants.narrow(0, 0, rgb)?, constants.narrow(0, rgb, rgb)?);

        let mut outputs = Vec::with_capacity(splices.len());
        for splice in splices {
            let previous = match splice.overlap as usize {
                0 => None,
                overlap => Some(self.buffer(self.tail, (rgb, overlap, height, width), dtype)?),
            };
            let bytes = backend::empty((splice.keep as usize, height, width, rgb), DType::U8)?;

            for (run, frame) in splicer.splice(previous.as_ref(), &raster, splice)? {
                let length = run.dims().get(1).copied().unwrap_or(0);
                let out = bytes.narrow(0, frame as usize, length)?;
                backend::launch::<RgbBytes>(
                    &rgb::Descriptor {
                        dtype,
                        channels: rgb::CHANNELS,
                        block: rgb::BLOCK,
                        pdl: false,
                    },
                    rgb::Args {
                        input: backend::operand(&run)?,
                        mean: backend::operand(&mean)?,
                        std: backend::operand(&std)?,
                        out: backend::operand(&out)?,
                    },
                )?;
            }
            outputs.push(bytes);
        }
        Ok(outputs)
    }

    /// The track forward of `layout`: its waveform, interleaved into PCM.
    fn decode_track(&self, layout: &Layout) -> Result<Vec<Tensor>> {
        let track = layout
            .audio
            .ok_or_else(|| Error::Invariant("a track's layout states no track".into()))?;
        let latent = self.model.latent_type();
        let shape = (voxels(&track) as usize, track.channels as usize);
        let latents = self.buffer(0, shape, latent)?;

        let (samples, channels) = self.model.pcm_window(&track)?;
        let waveform = backend::empty((channels as usize, samples as usize), DType::F32)?;
        self.model.decode(layout, &latents, None, Some(&waveform))?;

        let pcm = backend::empty((samples as usize, PCM_CHANNELS as usize), DType::I16)?;
        backend::launch::<PcmInterleave>(
            &pcm::Descriptor {
                dtype: DType::F32,
                block: pcm::BLOCK,
                pdl: false,
            },
            pcm::Args {
                left: backend::operand(&waveform.get(0)?)?,
                right: backend::operand(&waveform.get(1)?)?,
                out: backend::operand(&pcm)?,
                samples: u32::try_from(samples).map_err(|_| {
                    Error::Invalid("a track's samples exceed a launch value".into())
                })?,
            },
        )?;
        Ok(vec![pcm])
    }
}

impl Runner for DecoderRunner {
    fn phases(&self) -> &[Phase] {
        &self.config.phases
    }

    fn buckets(&self) -> Vec<Bucket> {
        self.config.buckets.capture_order().copied().collect()
    }

    /// With a splicer, the splice of each clip bucket into each timeline the deployment serves:
    /// the clip's bucket stating the request it serves (module documentation).
    fn eager_buckets(&self) -> Vec<Bucket> {
        if self.splicer.is_none() {
            return Vec::new();
        }
        let clips = self.config.buckets.capture_order();

        clips
            .flat_map(|clip| {
                let timelines = self.config.timelines.iter();
                timelines.filter_map(|&(frames, _)| splicing(clip, frames))
            })
            .collect()
    }

    /// A clip bucket's forward returns its raster, a splice bucket's the bytes of its staged
    /// splice, or of every distinct splice of its timeline when none is staged, and a track
    /// bucket's its PCM.
    fn forward(&self, bucket: &Bucket) -> Result<Vec<Tensor>> {
        let Extent::Clip {
            frames,
            height,
            width,
        } = bucket.extent
        else {
            return self.decode_track(&self.layout(bucket)?);
        };

        match bucket.serves {
            Some((timeline, ..)) => self.splice_clip(timeline, (frames, height, width)),
            None => self.decode_clip(&self.layout(bucket)?, (frames, height, width)),
        }
    }

    /// A decode reads no input tensor: its latents are the runner's buffers.
    fn idle_inputs(&self, bucket: &Bucket, _columns: &mut [u8]) -> Result<()> {
        self.layout(bucket).map(|_| ())
    }

    fn prepare_inputs(&mut self, batch: &Batch, tables: &BlockTables) -> Result<Bucket> {
        let (bucket, _) = self.units(batch, tables)?;

        self.graphs.prepare_inputs(&*self, &bucket, |_| Ok(()))?;
        Ok(bucket)
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
        let launched = served?;
        ended?;

        let landed = Instant::now();
        let launched = launched.unwrap_or(landed);
        Ok(BatchOutput {
            calls: batch.requests.iter().map(|_| performed()).collect(),
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
    /// batch whose end covers them.
    fn copy(&mut self, copies: &[BlockCopy]) -> Result<()> {
        self.memory.copy(copies, self.stream.as_ref())
    }

    fn graphs(&self) -> &Graphs {
        &self.graphs
    }
}

/// The elements of a latent shape per channel: its frames, rows and columns.
pub(super) fn voxels(shape: &nn::LatentShape) -> u64 {
    u64::from(shape.frames) * u64::from(shape.height) * u64::from(shape.width)
}

/// The elements of a latent shape.
pub(super) fn elements(shape: &nn::LatentShape) -> u64 {
    u64::from(shape.channels) * voxels(shape)
}

/// The bucket of the splice of `clip`'s clips into a timeline of `frames` frames: the clip's
/// bucket stating the request it serves; `None` for a bucket of no clip.
fn splicing(clip: &Bucket, frames: u32) -> Option<Bucket> {
    match clip.extent {
        Extent::Clip { height, width, .. } => Some(clip.serving(frames, width, height)),
        _ => None,
    }
}

/// The copy of a unit's output into the product at `at`: the first of the `outputs` of the
/// unit's last forward, its splice's bytes, or its decode's raster or PCM.
fn after(outputs: &[Tensor], at: u64) -> Result<PitchedBatch> {
    let kept = outputs
        .first()
        .ok_or_else(|| Error::Invariant("the unit's forward returned no output".into()))?;
    let bytes = kept.elem_count() as u64 * element_bytes(kept.dtype())?;

    let mut copies = PitchedBatch::new(SourceOrder::Stream);
    copies.push(pitched((at, bytes), (address(kept)?, bytes), bytes, 1)?)?;
    Ok(copies)
}

/// The copy of `rows` rows of `width` bytes, a row the destination's pitch after the one before
/// it in the destination and the source's pitch in the source.
pub(super) fn pitched(
    (destination, destination_pitch): (u64, u64),
    (source, source_pitch): (u64, u64),
    width: u64,
    rows: u32,
) -> Result<Pitched> {
    let size = |bytes: u64| {
        usize::try_from(bytes).map_err(|_| Error::Invalid("a copy exceeds memory".into()))
    };

    Ok(Pitched {
        destination,
        destination_pitch: size(destination_pitch)?,
        destination_layer: 1,
        source,
        source_pitch: size(source_pitch)?,
        source_layer: 1,
        width: size(width)?,
        height: rows as usize,
        depth: 1,
    })
}

#[cfg(test)]
mod tests;

//! `DecoderRunner`: the runner of every decoder, a video decoder with its splicer and an audio
//! decoder alike.
//!
//! The runner holds its model through `nn::Decoder`, and a video decoder's splicer through
//! `nn::Splicer`; it names no concrete model. A media row names the products its call reads and
//! writes as positions in the request's product table (`crate::batch::MediaRow::results`), and the
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
//!    against then copies the raster's last frames into its tail block, and exchanges tails with
//!    the ranks that decode its neighbours in the call ("The tail a unit fades against").
//! 2. **The splice** of a clip a splicer places. A clip whose splice fades against its
//!    predecessor has the predecessor's tail in the runner's tail buffer, copied from this rank's
//!    run or received from another rank, and the splice runs eagerly: the raster placed under the
//!    unit's splice and packed into bytes.
//! 3. The unit's output is copied into its product: the bytes of its splice into the video units
//!    block from the splice's first frame, the raster of a clip no splicer places, or the PCM into
//!    the audio samples block.
//!
//! Every address is computed and every block checked before anything is issued: a position the
//! tables hold no entry for, a write to a read-only entry, or bytes past a block refuse the
//! batch. The copies are the driver's, so the forwards' launches and the tails' exchanges are the
//! only kernels of a call.
//!
//! ## Why a clip's splice runs eagerly
//!
//! A splice's runs, their number and their frames, follow from the clip's place in its timeline
//! (`Splicer::splice`), which is a request's value, and a captured graph's launches follow from
//! its bucket alone: a graph would place the clip under every distinct splice of the deployment's
//! timelines to keep one. The splice runs eagerly instead, as the forward of a splice bucket
//! ([`Forward::eager_buckets`]): a clip's bucket stating the timeline it serves, whose forward
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
//! ## The deal
//!
//! A round whose row names no rank is dealt in shares of [`RoundConfig::units`] consecutive
//! units, the round's `i`-th share to the `i`-th of [`RoundConfig::ranks`] when the round's index
//! in the timeline is even and to the `i`-th from the last when it is odd
//! ([`RoundConfig::rank_of`]). Every round but the last is whole, so the rank that decodes the
//! last unit of a round decodes the first unit of the round after it. The media host finds each
//! unit it encodes by the same deal, in the run of the rank that decoded it.
//!
//! ## The tail a unit fades against
//!
//! Each rank writes the tail of a unit it decodes into its own run of the unit's tail block. The
//! unit after it reads that tail in one of three ways:
//!
//! - **The same rank, the same call.** The unit after it on the rank's stream reads the run the
//!   stream wrote.
//! - **Another rank, the same call.** The first unit of a rank's share follows the last of the
//!   share dealt before it, which another rank decodes. That rank sends the tail from its run of
//!   the tail block, behind the tail's copy, which follows the decode alone, and this rank
//!   receives it into its tail buffer behind its own unit's decode: one point-to-point exchange
//!   over the decoder ranks' communicator per pair of consecutive shares, the same between two
//!   ranks of one process as between two hosts. A unit that both receives a tail and sends one
//!   (a share of one unit) does both in one exchange. The receiving stream waits on the device
//!   for the sent tail, and each rank's host waits for its stream once it issued an exchange, as
//!   after any collective outside a capture (`DecoderRunner::exchange`): the ranks of a round
//!   decode their first units at once, and a splice waits for its predecessor's tail, which
//!   follows that unit's decode on its rank's stream.
//! - **An earlier call.** The first unit of a round reads the tail of the round before it, which
//!   the scheduler gives it only once that round completed, and `execute` returns only once its
//!   stream passed every copy: the write is ordered before the read by completion. The deal puts
//!   that tail in the rank's own run, where the round before it left it, and a row that names a
//!   rank reads it from that rank's own run too: no call reads another rank's run of a tail.

use std::cell::Cell;
use std::sync::Arc;
use std::time::Instant;

use crate::batch::{Batch, BatchOutput, Extent, MediaRow, Phase, TimingCounters};
use nn::{Clip, Decoder, Splice, Splicer};
use ops::pcm::{self, PcmInterleave};
use ops::rgb::{self, RgbBytes};
use platform::{
    Communicator, DataType, DevicePtr, DeviceRange, Event, EventFlags, Pitched, PitchedBatch,
    SourceOrder, Stream,
};
use proto::video::Layout;
use tensor::{DType, Device, Shape, Tensor};

use crate::memory::allocation::Storage;
use crate::memory::buffers::{BufferMemory, UNIT_BYTES};
use crate::memory::lend;
use crate::memory::tables::{BlockCopy, BlockTables};
use crate::runner::buckets::{Bucket, Buckets};
use crate::runner::graphs::Graphs;
use crate::runner::{Forward, Runner};
use crate::runner::{address, element_bytes, performed, settle};
use crate::worker::{Error, Result};

/// The channels of a decoded raster and of its packed bytes: red, green and blue.
const CHANNELS: u64 = rgb::CHANNELS as u64;

/// The element type a tail travels as between the decoder ranks: bytes, so its element count is
/// its bytes whatever the decoder's element type.
const TAIL_ELEMENT: DataType = DataType::Uint8;

/// What one unit of a call issues, in the order `serve` issues it (module documentation):
///
/// - the copies before its decode: the clip's window or the track into the latent buffer;
/// - the tail it leaves its successor: the copy of the raster's last frames into its tail block
///   behind the decode, and, when another rank decodes the successor in this call, the tail's
///   address in that block, the member of the decoder ranks' communicator it is sent to, and its
///   bytes;
/// - its splice: when another rank decoded its predecessor in this call, the member that sends
///   the predecessor's tail into the tail buffer and its bytes; otherwise the copy of the
///   predecessor's tail into the tail buffer; and the splice it places, `None` for a unit whose
///   decode is its output;
/// - the address in the product buffer its output lands at.
type Unit = (
    PitchedBatch,
    (PitchedBatch, Option<(u64, u32, u64)>),
    (Option<(u32, u64)>, PitchedBatch, Option<Splice>),
    u64,
);

/// What a deployment fixes of a decoder runner. Immutable once the runner is built.
#[derive(Clone, Debug)]
pub(crate) struct DecoderConfig {
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
/// decoders deal each round's shares to the ranks in order or in reverse order, alternately
/// (module documentation), and the media host finds each unit it encodes in the run of the rank
/// that decoded it. Immutable once a runner is built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RoundConfig {
    /// The ranks a round's units are dealt to, in the order of an even round, each by its rank
    /// index in the process's media memory (`BufferMemory::rank`). A rank's place here is its
    /// member index in the decoder ranks' communicator.
    pub ranks: Vec<u32>,
    /// The consecutive units each rank decodes in a round.
    pub units: u32,
}

impl RoundConfig {
    /// The units of one round: `units` for each of `ranks`.
    pub(crate) fn width(&self) -> u32 {
        let ranks = u32::try_from(self.ranks.len()).unwrap_or(u32::MAX);

        ranks.saturating_mul(self.units)
    }

    /// The rank that decodes unit `unit` of a timeline: the rank its round deals the share that
    /// holds it, round `k`'s `i`-th share going to the `i`-th of `ranks` for an even `k` and to
    /// the `i`-th from the last for an odd one. So the first unit of round `k + 1` falls to the
    /// rank of the last unit of round `k`, whole as every round but the last is. `None` for a
    /// deal of no unit.
    pub(crate) fn rank_of(&self, unit: u32) -> Option<u32> {
        let round = unit.checked_div(self.width())?;
        let share = unit.checked_rem(self.width())?.checked_div(self.units)? as usize;
        let place = match round % 2 {
            0 => share,
            _ => self.ranks.len().checked_sub(share + 1)?,
        };

        self.ranks.get(place).copied()
    }

    /// The place of rank `rank` among `ranks`: its member index in the decoder ranks'
    /// communicator. `None` for a rank no round is dealt to.
    pub(crate) fn member(&self, rank: u32) -> Option<u32> {
        let place = self.ranks.iter().position(|&dealt| dealt == rank)?;

        u32::try_from(place).ok()
    }

    /// Whether the configuration deals a round: at least one unit to each rank, and each rank
    /// once.
    pub(crate) fn deals(&self) -> bool {
        let once = (0..self.ranks.len()).all(|at| !self.ranks[..at].contains(&self.ranks[at]));

        self.units > 0 && !self.ranks.is_empty() && once
    }
}

/// The runner of a decoder (module documentation).
pub(crate) struct DecoderRunner {
    model: Arc<dyn Decoder + Send + Sync>,
    /// A video decoder's splicer; `None` for a track's decoder, or a clip's that places none.
    splicer: Option<Arc<dyn Splicer + Send + Sync>>,
    config: DecoderConfig,
    graphs: Graphs,
    /// The rank's media memory, which every media runner of the rank shares.
    memory: Arc<BufferMemory>,
    /// The stream the graphs replay on; `None` for graphs that hold no device.
    stream: Option<Stream>,
    /// A video decoder's communicator with the other ranks its rounds are dealt to, its member
    /// index the rank's place among them ([`RoundConfig::member`]), over which a round's tails
    /// pass between consecutive shares on `stream`; `None` for graphs that hold no device and
    /// for a round of one rank. The runner alone issues on it, from its worker's thread.
    communicator: Option<Communicator>,
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
    /// - `communicator` is the decoder ranks' communicator a video decoder on a device whose
    ///   rounds are dealt to more than one rank exchanges its tails over; `None` for any other
    ///   runner, which exchanges none.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for no bucket, a bucket that is neither a clip nor a track, a clip's
    /// that states a request (a splice's, [`Forward::eager_buckets`]), a video
    /// decoder serving no timeline or with a round that deals no unit to its memory's rank, a
    /// splicer of another element type, a track of other channels than a track encode takes, a
    /// stream without a device or the reverse; a communicator a runner on a device whose rounds
    /// pass tails between ranks lacks, or one another runner holds or whose members are not the
    /// round's ranks with this rank at its place; what the model or the splicer refuses: a canvas the model lays out no
    /// clip of, a timeline the splicer has no splices of; the driver's refusal of the buffers.
    pub(crate) fn new(
        model: Arc<dyn Decoder + Send + Sync>,
        splicer: Option<Arc<dyn Splicer + Send + Sync>>,
        config: DecoderConfig,
        graphs: Graphs,
        (stream, communicator): (Option<Stream>, Option<Communicator>),
        memory: Arc<BufferMemory>,
    ) -> Result<Self> {
        if graphs.enter().is_ok() != stream.is_some() || config.buckets.is_empty() {
            return Err(Error::Invalid(
                "a decoder runner takes buckets, and the stream of its graphs or none for graphs \
                 that hold no device"
                    .into(),
            ));
        }

        // A clip decoder on a device whose rounds go to several ranks passes tails between them
        // over a communicator of exactly those ranks, at this rank's place; no other runner
        // holds one.
        let clips = config
            .buckets
            .iter()
            .any(|bucket| matches!(bucket.extent, Extent::Clip { .. }));
        let exchanges = clips && stream.is_some() && config.round.ranks.len() > 1;
        let joined = communicator.as_ref().map(|communicator| {
            let members = usize::try_from(communicator.size()).ok();
            let place = u32::try_from(communicator.rank()).ok();
            members == Some(config.round.ranks.len()) && place == config.round.member(memory.rank())
        });
        if joined != exchanges.then_some(true) {
            return Err(Error::Invalid(format!(
                "the decoder of rank {} deals its rounds as {:?}, {} a device, and holds {}: a \
                 clip decoder on a device whose rounds go to several ranks passes its tails over \
                 a communicator of exactly those ranks at this rank's place, and no other \
                 decoder holds one",
                memory.rank(),
                config.round,
                if stream.is_some() { "on" } else { "on no" },
                match joined {
                    Some(true) => "such a communicator",
                    Some(false) => "a communicator of other members or another place",
                    None => "no communicator",
                },
            )));
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
                // A clip's bucket that states a request is a splice's (`Forward::eager_buckets`).
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
                    // The interleaved PCM is what the track encode takes, whose channels the
                    // media crate states: a waveform of other channels has no encode.
                    let channels = model.pcm_window(&track)?.1;
                    if channels != media::codec::PCM_CHANNELS {
                        return Err(Error::Invalid(format!(
                            "bucket {bucket}'s track has {channels} channels, and a track encode \
                             takes {}",
                            media::codec::PCM_CHANNELS
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
            communicator,
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
    pub(crate) const fn device_bytes(&self) -> u64 {
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
        // worker hands the batch to alone; otherwise the round's deal, whose rounds start
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
        // The member of the decoder ranks' communicator another rank of the deal is.
        let member = |rank: u32| {
            round.member(rank).ok_or_else(|| {
                Error::Invariant(format!(
                    "rank {rank} is none of the ranks {:?} a round is dealt to",
                    round.ranks
                ))
            })
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
            // read where the module documentation states by the rank that decodes its
            // predecessor: from this rank's run of its block, or received from another rank of
            // this call straight into the tail buffer.
            let (mut received, mut fade) = (None, copies());
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
                if before_rank == own {
                    let read = (tail_class, previous);
                    let source = self.place(tables, row, read, false, (0, bytes))?;
                    fade.push(pitched(
                        (base + self.tail, bytes),
                        (source, bytes),
                        bytes,
                        1,
                    )?)?;
                } else if index > 0 {
                    received = Some((member(before_rank)?, bytes));
                } else {
                    return Err(Error::Invariant(format!(
                        "rank {before_rank} decoded the unit before the round from unit {unit}, \
                         which the deal gives rank {own}: a round's first unit falls to the \
                         rank of the last unit of the round before it"
                    )));
                }
            }

            // The bytes of its own splice, from the splice's first frame of the timeline.
            let at = (
                u64::from(splice.first) * pixels * CHANNELS,
                u64::from(splice.keep) * pixels * CHANNELS,
            );
            let kept = self.place(tables, row, (media_class, media), true, at)?;

            // The tail it leaves its successor: the end of each channel's frames of the raster,
            // `[3, frames, rows, columns]`, packed into its tail block, from which it is sent to
            // the successor's rank when another rank decodes it in this call; one the next round
            // reads stays where it lies, which the deal makes that round's first rank's run.
            let (mut left, mut sent) = (copies(), None);
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
                    sent = Some((at, member(reader)?, bytes));
                }
            }
            let splice = Some(Splice { first: 0, ..splice });
            units.push((before, (left, sent), (received, fade, splice), kept));
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

    /// Exchanges a unit's tails with the ranks that decode its neighbours in the call, on the
    /// runner's stream over the decoder ranks' communicator (module documentation): `sent`, the
    /// address of the tail the unit left in this rank's run of its tail block, the member that
    /// decodes its successor and the tail's bytes; `received`, the member that decoded its
    /// predecessor and the bytes of the tail that member sends into the tail buffer. Both go in
    /// one NCCL group, so a rank that does both waits for neither peer before the other.
    ///
    /// A send and a receive wait on the device for their peer. A first launch of a kernel takes
    /// a lock of the whole process and waits for its device to drain, and the decoder ranks of a
    /// host are threads of one process, so a rank that went on to its splice's eager launches
    /// could hold that lock while its peer, waiting for it, queues on the lock in front of its
    /// own send (`ops::collective::settle`): the stream is therefore waited for here. A runner
    /// whose graphs hold no device states the exchange and issues nothing.
    ///
    /// # Safety
    ///
    /// For a runner on a device, `sent`'s `bytes` from its address lie in device memory that
    /// stays unwritten until the stream has passed the exchange.
    unsafe fn exchange(
        &self,
        sent: Option<(u64, u32, u64)>,
        received: Option<(u32, u64)>,
    ) -> Result<()> {
        if sent.is_none() && received.is_none() {
            return Ok(());
        }
        let Some(stream) = &self.stream else {
            return Ok(());
        };
        let communicator = self.communicator.as_ref().ok_or_else(|| {
            Error::Invariant(
                "a decoder that passes tails between ranks holds no communicator".into(),
            )
        })?;
        // Each side's range, its bytes and its peer, as NCCL takes them: the sent tail from its
        // block, the received one into the tail buffer, which holds the largest tail of the
        // deployment's splices (`new`).
        let tail = self.buffers.as_ref().map_or(0, |storage| storage.address()) + self.tail;
        let side = |address: u64, member: u32, bytes: u64| -> Result<(DeviceRange, usize, i32)> {
            let length = usize::try_from(bytes)
                .map_err(|_| Error::Invalid("a tail exceeds memory".into()))?;
            let peer = i32::try_from(member)
                .map_err(|_| Error::Invalid(format!("member {member} is past NCCL's ranks")))?;
            let range = DeviceRange::new(DevicePtr::from_address(address), length);
            Ok((range, length, peer))
        };
        let sent = sent
            .map(|(at, to, bytes)| side(at, to, bytes))
            .transpose()?;
        let received = received
            .map(|(from, bytes)| side(tail, from, bytes))
            .transpose()?;

        platform::nccl::group(|| {
            if let Some((range, count, peer)) = sent {
                communicator.send(stream, range, count, TAIL_ELEMENT, peer)?;
            }
            if let Some((range, count, peer)) = received {
                communicator.recv(stream, range, count, TAIL_ELEMENT, peer)?;
            }
            Ok(())
        })?;
        Ok(stream.synchronize()?)
    }

    /// Serves `batch`: every unit's decode, tail and splice on the stream, with the exchanges of
    /// the tails between the ranks (module documentation). Returns when the units were issued,
    /// for a batch that names a call. A runner whose graphs hold no device issues no copy, and
    /// its graphs refuse the first replay.
    fn serve(&self, batch: &Batch, tables: &BlockTables) -> Result<Option<Instant>> {
        // A batch that names no call carries copies alone.
        if batch.requests.is_empty() {
            return Ok(None);
        }
        let (bucket, units) = self.units(batch, tables)?;
        let frames = batch.media.first().map_or(0, |row| row.frames);

        // Every copy and exchange below is safe for one reason: each range lies in a block the
        // tables give this call or in the runner's buffers (`units`), which stay mapped until
        // the batch ends, and a forward's outputs stay until the next forward. The stream orders
        // each copy and exchange after the forward before it, and a buffer's next write after
        // the forward that read it; a tail block this call wrote is written by no later call.
        let launched = Instant::now();
        for (before, (left, sent), (received, fade, splice), kept) in &units {
            // The decode reads the unit's latents alone, so it waits for no other rank; the
            // exchange follows the tail it leaves.
            self.graphs.prepare_inputs(self, &bucket, |_| Ok(()))?;
            // SAFETY: as stated above the loop.
            unsafe { self.submit(before) }?;
            let mut outputs = self.graphs.replay(&bucket)?;

            // SAFETY: as stated above the loop.
            unsafe { self.submit(left) }?;
            // SAFETY: as stated above the loop.
            unsafe { self.exchange(*sent, *received) }?;

            // The splice fades against its predecessor's tail, in the tail buffer once the
            // exchange received it or the copy brought it from this rank's run.
            if let Some(splice) = *splice {
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
        let block = ops::empty((rows, channels), latent)?;
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
            let bytes = ops::empty((splice.keep as usize, height, width, rgb), DType::U8)?;

            for (run, frame) in splicer.splice(previous.as_ref(), &raster, splice)? {
                let length = run.dims().get(1).copied().unwrap_or(0);
                let out = bytes.narrow(0, frame as usize, length)?;
                ops::launch::<RgbBytes>(
                    &rgb::Descriptor::new(dtype, false),
                    rgb::Args {
                        input: ops::operand(&run)?,
                        mean: ops::operand(&mean)?,
                        std: ops::operand(&std)?,
                        out: ops::operand(&out)?,
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
        let waveform = ops::empty((channels as usize, samples as usize), DType::F32)?;
        self.model.decode(layout, &latents, None, Some(&waveform))?;

        let pcm = ops::empty((samples as usize, channels as usize), DType::I16)?;
        ops::launch::<PcmInterleave>(
            &pcm::Descriptor::new(DType::F32, false),
            pcm::Args {
                left: ops::operand(&waveform.get(0)?)?,
                right: ops::operand(&waveform.get(1)?)?,
                out: ops::operand(&pcm)?,
                samples: u32::try_from(samples).map_err(|_| {
                    Error::Invalid("a track's samples exceed a launch value".into())
                })?,
            },
        )?;
        Ok(vec![pcm])
    }
}

impl Forward for DecoderRunner {
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

    fn graphs(&self) -> &Graphs {
        &self.graphs
    }
}

impl Runner for DecoderRunner {
    fn phases(&self) -> &[Phase] {
        &self.config.phases
    }

    /// The decoder over `graphs`, over the runner's latent, tail and raster buffers when they
    /// lie beside the runner's graphs, and over buffers of its own for a rehearsal: a decode
    /// writes its tails and rasters there.
    fn capturer(&self, graphs: Graphs) -> Result<Box<dyn Forward>> {
        let buffers = crate::runner::instance_memory(&graphs, self.buffers.as_ref(), self.bytes)?;

        Ok(Box::new(Self {
            model: Arc::clone(&self.model),
            splicer: self.splicer.clone(),
            config: self.config.clone(),
            graphs,
            memory: Arc::clone(&self.memory),
            stream: None,
            communicator: None,
            done: None,
            layouts: self.layouts.clone(),
            staged: Cell::new(None),
            buffers,
            tail: self.tail,
            raster: self.raster,
            bytes: self.bytes,
        }))
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

    /// The rank's media memory: its product buffer and its parts of host regions
    /// (`BufferMemory::holds`).
    fn maps(&mut self, region: u8) -> bool {
        self.memory.holds(region)
    }

    /// The copies were issued on the runner's stream, which this waits for.
    fn wait_copies(&mut self) -> Result<()> {
        settle(self.stream.as_ref())
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

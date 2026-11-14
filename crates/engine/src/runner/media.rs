//! `MediaHost`: the runner of a video request's host work, the four phases no device computes:
//! reading its condition media (`MediaRead`), encoding each decoded round of its video and its
//! audio track (`VideoEncode`, `AudioEncode`), and muxing the encoded units and the track into
//! the request's MP4 (`Mux`).
//!
//! ## A batch
//!
//! A host batch takes two turns of its rank's worker ([`Runner::start`]); between them its work
//! runs on the process's media thread pool, and the worker serves other jobs:
//!
//! 1. `start` finds the blocks each row names in the rank's device block tables, by the row's
//!    slot, the class of each product kind and the row's positions (`MediaRow::results`), copies
//!    what the row reads to host memory on the runner's stream, from the rank's own run of the
//!    block or, where it backs none, another rank's of its process (a round's decoded units from
//!    the run of the rank that decoded each), waits for the copy, and hands the row's work to the
//!    pool;
//! 2. once every row's work ended, `execute` copies what each row produced into every run of the
//!    blocks it writes that the process maps, so each backing rank of the process holds it,
//!    waits for the copies, and answers each row.
//!
//! A host batch is run by one rank of its group in each process that runs it
//! (`crate::executor::Route`): a video encode in every process that holds ranks of its group,
//! each encoding the units its own ranks decoded, and every other host batch once. Only its
//! writes reach the other ranks' runs: a block a call writes is written in every run of it the
//! process maps, and no other block is touched.
//!
//! Every block is found and checked in the first turn: a position the tables hold no entry for,
//! a write to a read-only entry, or bytes past a block refuse the batch before any work starts.
//! The second turn reads no table: the batch's number stays open until it ends, so the buffer pool
//! gives none of the blocks up meanwhile, whatever the tables were told.
//!
//! A row whose work fails is answered with an error, and the batch's other rows stand. A batch
//! with no rows ends the copies that started with it and answers nothing.
//!
//! ## What each phase reads and writes
//!
//! | Phase | Reads | Writes |
//! | --- | --- | --- |
//! | `MediaRead` | the request's condition media, published in shared memory | condition pixels, condition samples and vision pixels, those the request's conditions call for, in that order |
//! | `VideoEncode` | the frames of the units of its round that its process's ranks decoded, from the video units block's run on the rank that decoded each | each such unit's encoding into its own encoded units block, in the runs of that block the process maps |
//! | `AudioEncode` | the video's duration of the decoded track | nothing: the encoded track joins the request's assembly |
//! | `Mux`, handing rounds over | the encoded units blocks of the next rounds' units, as far as each framed unit reaches | nothing: the units join the assembly |
//! | `Mux`, final | nothing | nothing: the call's output carries the artifact |
//!
//! A round is the width of [`HostConfig::round`] in units of the timeline: a video encode covers
//! the round from its row's `unit`, and a muxing call hands over its row's `rounds` rounds after
//! the units the assembly holds. The decoder group's ranks decode a round's units by that deal,
//! each into its own run of the video units block, so the encode of each process reads the
//! frames of the units its ranks decoded there, and a unit's encoded block is written in the
//! process that decoded it, from which a crossing carries it to the muxer's when the two differ.
//!
//! ## What the blocks hold
//!
//! | Product | Bytes |
//! | --- | --- |
//! | Condition pixels | every visual condition's pixels as the model's encoders read them (`MediaFormat::pixels`): `[3, frames, rows, columns]` FP32, one condition after another in request order |
//! | Condition samples | every audio track as `[2, 1, samples]` FP32 padded to whole latent hops, one after another |
//! | Vision pixels | every vision block's patch rows in the tower's element type, one after another |
//! | Video units | the timeline's frames, `[frames, rows, columns, 3]` bytes, unit `u`'s after those of every earlier unit (the splicer's placement, `runners::decoder`) |
//! | Encoded units | one block per unit, of the longest unit's `media::encoded_unit_bytes` bytes ([`MediaHost::encoded_bytes`]): the unit's encoding behind its length ([`media::frame_unit`]) from the first byte, so that each is written in the process that decoded its unit |
//! | Audio samples | `[samples, 2]` interleaved `i16` from the first byte |
//!
//! ## The assembly
//!
//! A request's MP4 is assembled in the process's `media::MediaRunner`, one session per request,
//! which the runner opens with the request's first audio encode or muxing call and its final
//! muxing call finishes: the units join it in timeline order, a round per muxing call, and the
//! track whenever its encode ran. A request's audio encode and muxing calls are therefore one
//! media host's. A session a request left unfinished is closed when another request takes its
//! slot.

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use crate::batch::product::{
    AUDIO_SAMPLES, CONDITION_PIXELS, CONDITION_SAMPLES, ENCODED_UNITS, VIDEO_UNITS, VISION_PIXELS,
};
use crate::batch::{
    Batch, BatchId, BatchOutput, CallOutput, CallStatus, ErrorCode, MediaOutput, MediaRow, Phase,
    TimingCounters, Variant,
};
use nn::capability::MediaFormat;
use ops::Generation;
use platform::Stream;
use storage::runtime::Run;
use tensor::Tensor;

use crate::memory::buffers::{BufferMemory, UNIT_BYTES};
use crate::memory::tables::{BlockCopy, BlockTables};
use crate::runner::Runner;
use crate::runner::buckets::{Bucket, Buckets};
use crate::runner::conditions;
use crate::runner::decoder::RoundConfig;
use crate::runner::graphs::Graphs;
use crate::runner::thread_pool::{HostAction, HostTask, ThreadPool};
use crate::runner::{performed, settle};
use crate::worker::completion::{Outcome, Waker};
use crate::worker::{Error, Result};

/// The bytes of the length an encoded unit's row starts with ([`media::frame_unit`]).
const LENGTH: u64 = size_of::<u64>() as u64;

/// The bytes of one stereo sample of a decoded track: two `i16`.
const STEREO: u64 = 2 * size_of::<i16>() as u64;

/// The bytes of one element of a condition's pixels and samples: FP32.
const FLOAT: u64 = size_of::<f32>() as u64;

/// Bytes of the media memory a rank reaches that a host call reads or writes: the region that
/// holds them, their address ([`BufferMemory::address`]: a device address, or for memory that
/// holds no device their offset from the region's first byte), and how many there are.
pub(crate) type Span = (u8, u64, u64);

/// What one row of a host batch moves: the spans it reads, one after another into its work's
/// input, and for each output its work produces, the spans the output is written to, one in every
/// run of its block that the rank's process maps.
pub(crate) type Transfers = (Vec<Span>, Vec<Vec<Span>>);

/// What one row's host work produced: the bytes of each output the row writes, in order and each
/// at most its spans' bytes, and the artifact a final muxing call published.
pub(crate) type Produced = (Vec<Vec<u8>>, Option<MediaOutput>);

/// What a deployment fixes of a media host. Immutable once the runner is built.
#[derive(Clone, Debug)]
pub(crate) struct HostConfig {
    /// The host phases the runner serves, as the scheduler names them in a batch: `MediaRead`,
    /// `VideoEncode`, `AudioEncode` and `Mux`, or some of them.
    pub phases: Vec<Phase>,
    /// The class of each product kind the runner's phases read or write, by its name in
    /// `crate::batch::product`.
    pub classes: Vec<(&'static str, u8)>,
    /// Each timeline the deployment serves: its output frames, and the frames each of its media
    /// units keeps, in timeline order (the model's `VideoModel::video_units`).
    pub timelines: Vec<(u32, Vec<u32>)>,
    /// How a timeline's decode rounds are dealt to the decoder group's ranks, as the decoders
    /// state it: a round's width is what a video encode covers from its row's `unit`, and what a
    /// muxing call hands over per round its row states; the deal is the rank in whose run the
    /// encode finds each unit's frames.
    pub round: RoundConfig,
    /// Frames per second of the artifact: the model's.
    pub frame_rate: u32,
    /// Samples per second of the track: the model's.
    pub audio_rate: u32,
    /// How the model's encoders read decoded condition media.
    pub format: MediaFormat,
}

/// The runner of a video request's host work (module documentation).
pub(crate) struct MediaHost {
    config: HostConfig,
    /// One host bucket per phase served: what a batch's shape is checked against.
    buckets: Buckets,
    /// Graphs that hold no device: the runner captures nothing and launches nothing.
    graphs: Graphs,
    /// The rank's media memory, which every media runner of the rank shares.
    memory: Arc<BufferMemory>,
    /// The stream the runner's copies are issued on; `None` for memory that holds no device.
    stream: Option<Stream>,
    /// The process's thread pool for media work.
    pool: Arc<ThreadPool>,
    /// The process's encoders, muxers and readers, and the assemblies under way.
    media: Arc<media::MediaRunner>,
    /// The assemblies the runner opened, by slot: the request whose MP4 is assembled and the
    /// units handed to it so far.
    assemblies: Vec<(u32, proto::request::RequestId, u32)>,
    /// The batches whose host work runs: when the work started, and each row's.
    running: Vec<(BatchId, Instant, Vec<Running>)>,
}

/// A row whose host work runs: its task on the thread pool, and the spans of each output it
/// writes.
type Running = (Arc<HostTask<MediaWork>>, Vec<Vec<Span>>);

/// One row's host work, as the media host's thread pool runs it: what it reads is in host
/// memory already, and the runner writes what it produces once it ended.
pub(crate) struct MediaWork {
    work: Box<dyn Fn() -> Result<Produced> + Send + Sync>,
}

impl HostAction for MediaWork {
    type Output = Produced;

    fn run(&self) -> Result<Produced> {
        (self.work)()
    }

    /// Nothing to release: what the work reads is its own host memory.
    fn release(&self) -> Result<()> {
        Ok(())
    }

    /// The bytes the work reads were copied to the host, and the copy waited for, before the work
    /// was submitted: no device work addresses them.
    fn input_status(&self) -> Outcome {
        Outcome::Success(())
    }
}

impl MediaHost {
    /// The runner of `config`'s host phases over `memory`, the rank's media memory, issuing its
    /// copies on `stream` (`None` exactly when the memory holds no device), running its work on
    /// `pool` and assembling artifacts in `media`. `kernels` is the generation the rank's other
    /// runners serve, which the runner's graphs hold so that every runner of the rank switches
    /// generations together.
    ///
    /// The encoders the artifact needs are required here, before any request: a host whose
    /// codecs lack them cannot serve a video at all.
    ///
    /// # Errors
    ///
    /// A phase that is no host phase, a product kind of a phase without a class, a round that
    /// deals no unit or a rank twice, a stream stated for memory without a device or the reverse,
    /// or missing codecs.
    pub(crate) fn new(
        config: HostConfig,
        kernels: Arc<Generation>,
        memory: Arc<BufferMemory>,
        stream: Option<Stream>,
        pool: Arc<ThreadPool>,
        media: Arc<media::MediaRunner>,
    ) -> Result<Self> {
        let mut buckets = Vec::with_capacity(config.phases.len());
        for &phase in &config.phases {
            let variant = variant(phase).ok_or_else(|| {
                Error::Invalid(format!(
                    "phase {phase:?} is no host phase of a media request"
                ))
            })?;
            for kind in kinds(phase) {
                class(&config, kind)?;
            }
            buckets.push(Bucket::host(variant));
        }

        if !config.round.deals() {
            return Err(Error::Invalid(format!(
                "the decode rounds are dealt as {:?}, which is no deal of a unit to each of \
                 distinct ranks",
                config.round
            )));
        }
        if memory.device().is_some() != stream.is_some() {
            return Err(Error::Invalid(
                "a media host issues its copies on a stream exactly when its memory is on a \
                 device"
                    .into(),
            ));
        }
        if config.phases.iter().any(|phase| *phase != Phase::MediaRead) {
            media.require_codecs()?;
        }

        Ok(Self {
            buckets: Buckets::new(buckets),
            graphs: Graphs::meta(kernels, 0),
            config,
            memory,
            stream,
            pool,
            media,
            assemblies: Vec::new(),
            running: Vec::new(),
        })
    }

    /// The bytes of an encoded units block for a timeline whose units keep `units` frames at
    /// `height` by `width`: one unit's row, of the longest unit (module documentation). The
    /// scheduler sizes each unit's block by it.
    pub(crate) fn encoded_bytes(units: &[u32], height: u32, width: u32) -> u64 {
        (0..units.len())
            .map(|unit| encoded(units, unit..unit + 1, height, width))
            .max()
            .unwrap_or(0)
    }

    /// What each row of `batch` reads and writes, in row order, as the batch's first turn finds
    /// it in `tables` (module documentation): what `start` copies to the host for the row's work,
    /// and where `execute` copies what the work produced. A block is read from the rank's own run,
    /// or another rank's of its process where it backs none; a round's decoded units from the run
    /// of the rank that decoded each; an encoded unit's row as far as its framed unit reaches. An
    /// output is written to every run of its block.
    ///
    /// # Errors
    ///
    /// A batch of a phase or shape the runner does not serve; a row whose results do not match
    /// its phase; a position the tables hold no entry for, a write to an entry they state
    /// read-only, or bytes past a block; a timeline or a unit the deployment does not serve.
    pub(crate) fn transfers(&self, batch: &Batch, tables: &BlockTables) -> Result<Vec<Transfers>> {
        let phase = self.phase_of(batch)?;

        batch
            .media
            .iter()
            .map(|row| self.row_transfers(phase, row, tables))
            .collect()
    }

    /// The work of row `index` of `batch` over `input`, the bytes its reads hold one after
    /// another as [`MediaHost::transfers`] states them: what `start` hands the thread pool. A
    /// muxing call takes its rounds and an audio encode or a muxing call opens the request's
    /// assembly here, in batch order, on the worker's thread.
    ///
    /// # Errors
    ///
    /// As [`MediaHost::transfers`], a row the batch does not hold, or an assembly the media runner
    /// refuses to open.
    pub(crate) fn work(
        &mut self,
        batch: &Batch,
        index: usize,
        input: Vec<u8>,
    ) -> Result<MediaWork> {
        let phase = self.phase_of(batch)?;
        let row = batch.media.get(index).ok_or_else(|| {
            Error::Invalid(format!(
                "batch {} holds {} rows and no row {index}",
                batch.id.0,
                batch.media.len()
            ))
        })?;
        let media = Arc::clone(&self.media);
        let request = row.request.0;

        let work: Box<dyn Fn() -> Result<Produced> + Send + Sync> = match phase {
            Phase::MediaRead => {
                let conditioning = conditions_of(row)?;
                let format = self.config.format;
                let rates = (self.config.frame_rate, self.config.audio_rate);

                Box::new(move || {
                    let regions = read_conditions(&media, rates, &conditioning, &format)?;
                    Ok((regions, None))
                })
            }

            Phase::VideoEncode => {
                let units = self.units(row.frames)?;
                let round = self.round(units, row.unit, self.config.round.width())?;
                let frames: Vec<u32> = self
                    .decoded_here(round)
                    .into_iter()
                    .map(|unit| units[unit])
                    .collect();
                let (height, width, rate) = (row.height, row.width, self.config.frame_rate);

                Box::new(move || Ok((encode_units(&input, &frames, height, width, rate)?, None)))
            }

            Phase::AudioEncode => {
                let samples = self.track_samples(row)?;
                let rate = self.config.audio_rate;
                self.assemble(row)?;

                Box::new(move || {
                    discarding(&media, request, || {
                        let pcm: Vec<i16> = input
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|pair| i16::from_ne_bytes(*pair))
                            .collect();
                        let track = media.encode_audio(&pcm, samples, rate)?;
                        media.set_audio(request, track)?;
                        Ok((Vec::new(), None))
                    })
                })
            }

            Phase::Mux if row.rounds > 0 => {
                let handed = self.handed(row)?;
                let at = self.assemble(row)?;
                let count = handed.len();
                let taken = u32::try_from(count).unwrap_or(u32::MAX);
                self.assemblies[at].2 = self.assemblies[at].2.saturating_add(taken);

                Box::new(move || {
                    discarding(&media, request, || {
                        media.append_units(request, &framed_units(&input, count)?)?;
                        Ok((Vec::new(), None))
                    })
                })
            }

            // The final call: the assembly leaves the runner with it.
            Phase::Mux => {
                self.assemblies
                    .retain(|(slot, held, _)| (*slot, *held) != (row.slot, row.request));

                Box::new(move || {
                    discarding(&media, request, || {
                        let artifact = media.finish_mux(request)?;
                        Ok((Vec::new(), Some(publish(request, &artifact)?)))
                    })
                })
            }

            other => {
                return Err(Error::Invariant(format!(
                    "the media host serves host phases and was asked for {other:?}"
                )));
            }
        };

        Ok(MediaWork { work })
    }

    /// The phase of `batch`, which the runner serves in the variant the batch states.
    fn phase_of(&self, batch: &Batch) -> Result<Phase> {
        let bucket = self.buckets.require(&batch.shape)?;
        if variant(batch.phase) != Some(bucket.variant) {
            return Err(Error::Invalid(format!(
                "batch {} of phase {:?} states variant {}, which is not that phase's",
                batch.id.0, batch.phase, batch.shape.variant
            )));
        }

        Ok(batch.phase)
    }

    /// What `row`, a row of a batch of `phase`, reads and writes: see [`MediaHost::transfers`].
    fn row_transfers(
        &self,
        phase: Phase,
        row: &MediaRow,
        tables: &BlockTables,
    ) -> Result<Transfers> {
        let split = (row.reads as usize).min(row.results.len());
        let (reads, writes) = row.results.split_at(split);
        let mismatched = || {
            Error::Invalid(format!(
                "request {}'s row of phase {phase:?} reads {} results and writes {}, which the \
                 phase does not",
                row.request.0,
                reads.len(),
                writes.len()
            ))
        };

        match phase {
            Phase::MediaRead => {
                let extents = extents(&conditions_of(row)?, &self.config.format)?;
                if !reads.is_empty() || writes.len() != extents.len() {
                    return Err(mismatched());
                }

                let mut written = Vec::with_capacity(extents.len());
                for (&position, (kind, bytes)) in writes.iter().zip(extents) {
                    let runs = self.written(tables, row.slot, kind, position)?;
                    written.push(self.spans(&runs, 0, bytes)?);
                }
                Ok((Vec::new(), written))
            }

            Phase::VideoEncode => {
                let [frames_at] = reads else {
                    return Err(mismatched());
                };
                let units = self.units(row.frames)?;
                let round = self.round(units, row.unit, self.config.round.width())?;
                let (height, width) = (row.height, row.width);
                // One encoded units block per unit of the round, in unit order.
                if writes.len() != round.len() {
                    return Err(mismatched());
                }

                // The units this process's ranks decoded, each unit's frames from the run of the
                // rank that decoded them, one after another in timeline order; the units of the
                // round another process decoded are that process's to encode.
                let frame = u64::from(height) * u64::from(width) * 3;
                let decoded = tables.runs(row.slot, class(&self.config, VIDEO_UNITS)?, *frames_at);
                let here = self.decoded_here(round.clone());
                let read = here
                    .iter()
                    .map(|&unit| {
                        let rank = u32::try_from(unit)
                            .ok()
                            .and_then(|unit| self.config.round.rank_of(unit));
                        let run = rank.and_then(|rank| self.memory.run_on(decoded, rank));
                        let run = run.ok_or_else(|| {
                            Error::Invalid(format!(
                                "slot {} holds no video units block at position {frames_at} on \
                                 the rank of this process that decodes unit {unit}",
                                row.slot
                            ))
                        })?;

                        let before = frames(&units[..unit]) * frame;
                        self.span(run, before, u64::from(units[unit]) * frame)
                    })
                    .collect::<Result<Vec<_>>>()?;

                // Each unit encoded here into its own block's runs in this process.
                let rows = here
                    .iter()
                    .map(|&unit| {
                        let encoded_at = writes[unit - round.start];
                        let runs = self.written(tables, row.slot, ENCODED_UNITS, encoded_at)?;
                        self.spans(&runs, 0, encoded(units, unit..unit + 1, height, width))
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok((read, rows))
            }

            Phase::AudioEncode => {
                let ([samples_at], []) = (reads, writes) else {
                    return Err(mismatched());
                };

                // The video's duration of the track, which the decoder's window covers; a block
                // that holds less is read whole, and the track padded with silence.
                let run = self.read(tables, row.slot, AUDIO_SAMPLES, *samples_at)?;
                let duration = self.track_samples(row)? as u64 * STEREO;
                let read = self.span(run, 0, duration.min(u64::from(run.2) * UNIT_BYTES))?;
                Ok((vec![read], Vec::new()))
            }

            Phase::Mux => {
                // Every muxing call may name the encoded units it reads; only one that hands
                // rounds over reads them, the block of each unit it hands over in unit order.
                if !writes.is_empty() {
                    return Err(mismatched());
                }
                if row.rounds == 0 {
                    return Ok((Vec::new(), Vec::new()));
                }
                let handed = self.handed(row)?;
                if reads.len() != handed.len() {
                    return Err(mismatched());
                }

                let units = self.units(row.frames)?;
                let (height, width) = (row.height, row.width);
                let rows = handed
                    .zip(reads)
                    .map(|(unit, &encoded_at)| {
                        let run = self.read(tables, row.slot, ENCODED_UNITS, encoded_at)?;
                        self.span(run, 0, encoded(units, unit..unit + 1, height, width))
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok((rows, Vec::new()))
            }

            other => Err(Error::Invariant(format!(
                "the media host serves host phases and was asked for {other:?}"
            ))),
        }
    }

    /// The run a host batch reads the block `slot` holds at `position` of the class of product
    /// `kind` from: the rank's own, or where the rank backs none of the block another rank's of
    /// its process.
    fn read(&self, tables: &BlockTables, slot: u32, kind: &str, position: u32) -> Result<Run> {
        let class = class(&self.config, kind)?;
        let own = tables.run(slot, class, position);

        own.or_else(|| tables.runs(slot, class, position).first().copied())
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "slot {slot} holds no {kind} block at position {position} of class {class} \
                     on this rank's process"
                ))
            })
    }

    /// Every run of the block `slot` holds at `position` of the class of product `kind` that the
    /// rank's process maps, which a host batch writes.
    fn written(
        &self,
        tables: &BlockTables,
        slot: u32,
        kind: &str,
        position: u32,
    ) -> Result<Vec<Run>> {
        let class = class(&self.config, kind)?;
        if !tables.writable(slot, class, position) {
            return Err(Error::Invalid(format!(
                "a host batch writes the {kind} block of slot {slot} at position {position} of \
                 class {class}, which holds no entry or is read-only"
            )));
        }

        Ok(tables.runs(slot, class, position).to_vec())
    }

    /// The span of `bytes` bytes from byte `offset` of `run`.
    fn span(&self, run: Run, offset: u64, bytes: u64) -> Result<Span> {
        Ok((run.0, self.memory.address(run, offset, bytes)?, bytes))
    }

    /// The spans of `bytes` bytes from byte `offset` of each of `runs`.
    fn spans(&self, runs: &[Run], offset: u64, bytes: u64) -> Result<Vec<Span>> {
        runs.iter()
            .map(|&run| self.span(run, offset, bytes))
            .collect()
    }

    /// The frames each unit of a timeline of `frames` output frames keeps.
    fn units(&self, frames: u32) -> Result<&[u32]> {
        self.config
            .timelines
            .iter()
            .find(|(served, _)| *served == frames)
            .map(|(_, units)| units.as_slice())
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "the deployment serves no timeline of {frames} frames"
                ))
            })
    }

    /// The units of `units` from unit `first`, at most `count` of them.
    fn round(&self, units: &[u32], first: u32, count: u32) -> Result<Range<usize>> {
        let first = first as usize;
        if first >= units.len() || count == 0 {
            return Err(Error::Invalid(format!(
                "unit {first} is not one of the {} units of the request's timeline",
                units.len()
            )));
        }

        Ok(first..first.saturating_add(count as usize).min(units.len()))
    }

    /// The units of `round`, a video encode's, that ranks of this process decoded, in timeline
    /// order: those the decode rounds' deal gives to a rank whose product buffer the rank's
    /// memory reaches. The process encodes them, and each other process the units its own ranks
    /// decoded ([`crate::batch::Variant::dealt`]).
    fn decoded_here(&self, round: Range<usize>) -> Vec<usize> {
        round
            .filter(|&unit| {
                let rank = u32::try_from(unit)
                    .ok()
                    .and_then(|unit| self.config.round.rank_of(unit));
                rank.is_some_and(|rank| self.memory.reaches(rank))
            })
            .collect()
    }

    /// The units a muxing call of `row` hands over: its rounds after the units the request's
    /// assembly holds.
    fn handed(&self, row: &MediaRow) -> Result<Range<usize>> {
        let held = self
            .assemblies
            .iter()
            .find(|(slot, request, _)| (*slot, *request) == (row.slot, row.request))
            .map_or(0, |(.., handed)| *handed);
        let count = row.rounds.saturating_mul(self.config.round.width());

        self.round(self.units(row.frames)?, held, count)
    }

    /// The samples of `row`'s track: the video's duration at the track's rate.
    fn track_samples(&self, row: &MediaRow) -> Result<usize> {
        let frames = frames(self.units(row.frames)?);
        let frames = usize::try_from(frames)
            .map_err(|_| Error::Invalid("a timeline is longer than memory".into()))?;

        Ok(media::audio_samples(
            frames,
            self.config.frame_rate,
            self.config.audio_rate,
        ))
    }

    /// The assembly of `row`'s request among the runner's, opened in the media runner when the
    /// runner holds none for it: its index in `assemblies`. A session another request left on
    /// the row's slot is closed first.
    fn assemble(&mut self, row: &MediaRow) -> Result<usize> {
        if let Some(at) = self
            .assemblies
            .iter()
            .position(|(slot, ..)| *slot == row.slot)
        {
            if self.assemblies[at].1 == row.request {
                return Ok(at);
            }

            // The request before it on the slot left without its artifact.
            let (_, left, _) = self.assemblies.swap_remove(at);
            self.media.close_mux(left.0);
        }

        let units = self.units(row.frames)?;
        self.media.open_mux(
            row.request.0,
            row.width,
            row.height,
            self.config.frame_rate,
            self.config.audio_rate,
            units,
        )?;
        self.assemblies.push((row.slot, row.request, 0));
        Ok(self.assemblies.len() - 1)
    }

    /// The bytes of `spans` in host memory, one span after another; an encoded unit's row
    /// (`framed`) as far as its framed unit reaches. Memory that holds no device holds no bytes,
    /// and a read there finds zeros, as a meta tensor holds no values.
    fn fetch(&self, spans: &[Span], framed: bool) -> Result<Vec<u8>> {
        let Some(stream) = &self.stream else {
            let bytes = if framed {
                LENGTH * spans.len() as u64
            } else {
                spans.iter().map(|span| span.2).sum()
            };
            return Ok(vec![0; host_bytes(bytes)?]);
        };

        let lengths: Vec<u64> = if framed {
            let mut prefixes = vec![0_u8; host_bytes(LENGTH * spans.len() as u64)?];
            let reads: Vec<(u64, u64)> = spans.iter().map(|span| (span.1, LENGTH)).collect();
            copy_out(stream, &reads, &mut prefixes)?;

            spans
                .iter()
                .zip(prefixes.as_chunks::<{ size_of::<u64>() }>().0)
                .map(|(span, prefix)| {
                    // The length `media::frame_unit` framed the unit with.
                    LENGTH
                        .checked_add(u64::from_ne_bytes(*prefix))
                        .filter(|bytes| *bytes <= span.2)
                        .ok_or_else(|| {
                            Error::Invalid(
                                "an encoded unit's row names a length past the row".into(),
                            )
                        })
                })
                .collect::<Result<_>>()?
        } else {
            spans.iter().map(|span| span.2).collect()
        };

        let mut input = vec![0_u8; host_bytes(lengths.iter().sum())?];
        let reads: Vec<(u64, u64)> = spans
            .iter()
            .zip(&lengths)
            .map(|(span, bytes)| (span.1, *bytes))
            .collect();
        copy_out(stream, &reads, &mut input)?;
        Ok(input)
    }
}

impl Runner for MediaHost {
    fn phases(&self) -> &[Phase] {
        &self.config.phases
    }

    /// A host call captures no graph: its buckets state which phases the runner serves and are
    /// no warm-up's.
    fn buckets(&self) -> Vec<Bucket> {
        Vec::new()
    }

    fn forward(&self, bucket: &Bucket) -> Result<Vec<Tensor>> {
        Err(Error::Unsupported(format!(
            "bucket {bucket} is a host call's, which has no forward"
        )))
    }

    fn idle_inputs(&self, bucket: &Bucket, columns: &mut [u8]) -> Result<()> {
        let _ = columns;
        Err(Error::Unsupported(format!(
            "bucket {bucket} is a host call's, which has no input tensors"
        )))
    }

    fn prepare_inputs(&mut self, batch: &Batch, tables: &BlockTables) -> Result<Bucket> {
        let _ = tables;
        Err(Error::Unsupported(format!(
            "batch {} is a host call's, which has no input tensors",
            batch.id.0
        )))
    }

    /// Finds every block the batch's rows name, copies what they read to the host and hands each
    /// row's work to the thread pool (module documentation); `resume` is called once every
    /// row's work ended. A refused batch returns once the stream passed the copies that started
    /// with it.
    fn start(&mut self, batch: &Batch, tables: &BlockTables, resume: Waker) -> Result<bool> {
        if batch.media.is_empty() {
            return Ok(false);
        }

        let begun = self.begin(batch, tables, resume);
        if begun.is_err()
            && let Err(error) = settle(self.stream.as_ref())
        {
            tracing::error!(%error, "a refused host batch's copies cannot be awaited");
        }
        begun.map(|()| true)
    }

    /// Writes what each row's work produced into the blocks `start` found, and answers each row;
    /// a batch with no rows answers none. It returns once the stream passed the writes and the
    /// copies that started with the batch, also when it fails.
    fn execute(&mut self, batch: &Batch, tables: &BlockTables) -> Result<BatchOutput> {
        let _ = tables;
        let answered = if batch.media.is_empty() {
            Ok(BatchOutput::default())
        } else {
            self.answer(batch)
        };

        let settled = settle(self.stream.as_ref());
        let output = answered?;
        settled?;
        Ok(output)
    }

    fn graphs(&self) -> &Graphs {
        &self.graphs
    }

    /// Issues the copies that start with the batch through the rank's media memory, on the
    /// runner's stream; a copy of other memory, such as the KV cache's, is refused.
    fn copy(&mut self, copies: &[BlockCopy]) -> Result<()> {
        let foreign = copies.iter().find(|copy| {
            copy.runs
                .iter()
                .any(|(from, to)| !self.memory.holds(from.0) || !self.memory.holds(to.0))
        });
        if let Some(copy) = foreign {
            return Err(Error::Unsupported(format!(
                "request {} copies memory other than the rank's media memory, which the media \
                 host does not hold",
                copy.request.0
            )));
        }

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

/// The two turns of a host batch: see [`Runner::start`] and [`Runner::execute`].
impl MediaHost {
    /// The first turn of `batch`, a batch with rows: see [`Runner::start`].
    fn begin(&mut self, batch: &Batch, tables: &BlockTables, resume: Waker) -> Result<()> {
        let started = Instant::now();

        // Every block is found and checked before any work starts.
        let transfers = self.transfers(batch, tables)?;

        let mut rows = Vec::with_capacity(transfers.len());
        for (index, (reads, writes)) in transfers.into_iter().enumerate() {
            let input = self.fetch(&reads, batch.phase == Phase::Mux)?;
            let work = self.work(batch, index, input)?;
            rows.push((self.pool.submit(work)?, writes));
        }

        // The batch resumes with the last row's end, on the thread that ended it.
        let left = Arc::new(AtomicUsize::new(rows.len()));
        let wake: Waker = Arc::new(move || {
            if left.fetch_sub(1, Ordering::AcqRel) == 1 {
                resume();
            }
        });
        for (task, _) in &rows {
            task.on_settled(&wake);
        }

        self.running.push((batch.id, started, rows));
        Ok(())
    }

    /// The second turn of `batch`, a batch with rows whose work ended: writes what each row
    /// produced and answers each row.
    fn answer(&mut self, batch: &Batch) -> Result<BatchOutput> {
        let at = self
            .running
            .iter()
            .position(|(id, ..)| *id == batch.id)
            .ok_or_else(|| {
                Error::Invariant(format!(
                    "batch {} of phase {:?} ends with no host work started for it",
                    batch.id.0, batch.phase
                ))
            })?;
        let (_, started, rows) = self.running.swap_remove(at);

        let outcomes: Vec<Outcome<Arc<Produced>>> =
            rows.iter().map(|(task, _)| task.wait()).collect();

        // What each row writes is checked before any copy is issued.
        let mut writes = Vec::new();
        for ((_, outputs), outcome) in rows.iter().zip(&outcomes) {
            let Outcome::Success(produced) = outcome else {
                continue;
            };
            if produced.0.len() != outputs.len() {
                return Err(Error::Invariant(format!(
                    "a row of batch {} produced {} outputs for {}",
                    batch.id.0,
                    produced.0.len(),
                    outputs.len()
                )));
            }
            for (spans, bytes) in outputs.iter().zip(&produced.0) {
                for span in spans {
                    if bytes.len() as u64 > span.2 {
                        return Err(Error::Invariant(format!(
                            "a row of batch {} produced {} bytes for a span of {}",
                            batch.id.0,
                            bytes.len(),
                            span.2
                        )));
                    }
                    writes.push((span.1, bytes.as_slice()));
                }
            }
        }
        let issued = match &self.stream {
            Some(stream) => copy_in(stream, &writes),
            None => Ok(()),
        };

        // The writes end before the bytes they read are dropped, whether or not every write was
        // issued.
        let settled = settle(self.stream.as_ref());
        issued?;
        settled?;

        let calls = outcomes
            .iter()
            .zip(&batch.media)
            .map(|(outcome, row)| match outcome {
                Outcome::Success(produced) => CallOutput {
                    media_output: produced.1.clone(),
                    ..performed()
                },
                Outcome::Failed(error) => failed(batch.phase, row, &error.to_string()),
                Outcome::Cancelled => failed(
                    batch.phase,
                    row,
                    "the media thread pool withdrew the work before it ran",
                ),
            })
            .collect();

        Ok(BatchOutput {
            calls,
            timing: TimingCounters {
                host_us: u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
                ..TimingCounters::default()
            },
            forward_stats: None,
        })
    }
}

/// The variant a batch of host phase `phase` is launched as; `None` for any other phase.
const fn variant(phase: Phase) -> Option<Variant> {
    match phase {
        Phase::MediaRead => Some(Variant::MediaRead),
        Phase::VideoEncode => Some(Variant::VideoEncode),
        Phase::AudioEncode => Some(Variant::AudioEncode),
        Phase::Mux => Some(Variant::Mux),
        _ => None,
    }
}

/// The product kinds a batch of host phase `phase` may read or write.
const fn kinds(phase: Phase) -> &'static [&'static str] {
    match phase {
        Phase::MediaRead => &[CONDITION_PIXELS, CONDITION_SAMPLES, VISION_PIXELS],
        Phase::VideoEncode => &[VIDEO_UNITS, ENCODED_UNITS],
        Phase::AudioEncode => &[AUDIO_SAMPLES],
        Phase::Mux => &[ENCODED_UNITS],
        _ => &[],
    }
}

/// The class of product `kind` in `config`.
fn class(config: &HostConfig, kind: &str) -> Result<u8> {
    config
        .classes
        .iter()
        .find(|(name, _)| *name == kind)
        .map(|(_, class)| *class)
        .ok_or_else(|| Error::Invalid(format!("the media host states no class of {kind}")))
}

/// The frames `units` keep together.
fn frames(units: &[u32]) -> u64 {
    units.iter().map(|frames| u64::from(*frames)).sum()
}

/// The bytes of the encoded units rows of `units[range]` at `height` by `width`.
fn encoded(units: &[u32], range: Range<usize>, height: u32, width: u32) -> u64 {
    units[range]
        .iter()
        .map(|frames| media::encoded_unit_bytes(*frames, height, width) as u64)
        .sum()
}

/// `bytes` as a host allocation's length.
fn host_bytes(bytes: u64) -> Result<usize> {
    usize::try_from(bytes).map_err(|_| Error::Resource("a host call's bytes exceed memory"))
}

/// Copies each of `reads`, an address and a length, into `into`, one after another, on `stream`,
/// and waits for them.
fn copy_out(stream: &Stream, reads: &[(u64, u64)], into: &mut [u8]) -> Result<()> {
    let mut at = 0;
    let mut issued = Ok(());

    for &(address, bytes) in reads {
        let length = host_bytes(bytes)?;
        let Some(window) = into.get_mut(at..at + length) else {
            issued = Err(Error::Invariant(
                "a host read reaches past its buffer".into(),
            ));
            break;
        };

        // SAFETY: the source lies in the rank's media memory (`BufferMemory::address` checked
        // it), which the buffer pool keeps for the batch while its number is open; the
        // destination is `into`, which nothing else touches until the wait below returns.
        let copied =
            unsafe { platform::memcpy_async(window.as_mut_ptr() as u64, address, length, stream) };
        if let Err(error) = copied {
            issued = Err(error.into());
            break;
        }
        at += length;
    }

    // The buffer outlives every copy issued into it, whether or not all were.
    let settled = stream.synchronize();
    issued?;
    Ok(settled?)
}

/// Copies each of `writes`, a destination address and its bytes, on `stream`; the caller waits
/// for them before it drops the bytes.
fn copy_in(stream: &Stream, writes: &[(u64, &[u8])]) -> Result<()> {
    for &(address, bytes) in writes {
        // SAFETY: the destination lies in a block of the rank's media memory that `start` found
        // writable, which the buffer pool keeps for the batch while its number is open; the
        // source is a row's produced bytes, which the caller keeps until its stream has passed
        // the copy.
        unsafe { platform::memcpy_async(address, bytes.as_ptr() as u64, bytes.len(), stream) }?;
    }

    Ok(())
}

/// The answer to `row`, a row of a batch of `phase` whose work failed with `error`: a read of
/// condition media the request's input could not give, any other the computation's failure.
fn failed(phase: Phase, row: &MediaRow, error: &str) -> CallOutput {
    tracing::warn!(
        request = row.request.0,
        ?phase,
        error,
        "a media request's host work failed"
    );

    let (code, message) = match phase {
        Phase::MediaRead => (
            ErrorCode::InputError,
            Some(format!(
                "request {}'s media could not be read: {error}",
                row.request.0
            )),
        ),
        _ => (ErrorCode::ComputeError, None),
    };
    CallOutput {
        status: CallStatus::Error,
        error_code: Some(code),
        error_message: message,
        ..performed()
    }
}

/// Runs `work` for `request`'s assembly, closing the assembly when the work fails: the request
/// then ends without its artifact, and nothing else reaches its session.
fn discarding(
    media: &media::MediaRunner,
    request: u64,
    work: impl FnOnce() -> Result<Produced>,
) -> Result<Produced> {
    let produced = work();
    if produced.is_err() {
        media.close_mux(request);
    }

    produced
}

/// Publishes `request`'s `artifact` in shared memory for the process that receives the call's
/// output, which claims and unlinks it.
fn publish(request: u64, artifact: &[u8]) -> Result<MediaOutput> {
    let locator = proto::MediaSource::publish(artifact)
        .map_err(|error| {
            Error::Invalid(format!(
                "request {request}'s artifact could not be published: {error}"
            ))
        })?
        .into_locator();

    Ok(MediaOutput {
        handle: crate::batch::ArtifactHandle::PosixShm { name: locator.name },
        bytes: locator.bytes,
    })
}

/// Each unit of `frames` frames at `height` by `width`, taken from `input` one after another,
/// encoded and framed by its length ([`media::frame_unit`]).
fn encode_units(
    input: &[u8],
    frames: &[u32],
    height: u32,
    width: u32,
    rate: u32,
) -> Result<Vec<Vec<u8>>> {
    let frame = height as usize * width as usize * 3;
    let mut at = 0;
    let mut rows = Vec::with_capacity(frames.len());

    for &count in frames {
        let bytes = count as usize * frame;
        let pixels = input.get(at..at + bytes).ok_or_else(|| {
            Error::Invalid(format!(
                "a round's frames of {} bytes hold no unit of {count} frames at {width}x{height} \
                 from byte {at}",
                input.len()
            ))
        })?;
        at += bytes;

        let unit = media::encode_video_unit(
            media::Rgb24Frames {
                pixels,
                height,
                width,
            },
            rate,
        )?;
        let mut row = vec![0_u8; size_of::<u64>() + unit.len()];
        media::frame_unit(&unit, &mut row)?;
        rows.push(row);
    }
    Ok(rows)
}

/// The `count` framed units `input` holds one after another, each without its length.
fn framed_units(input: &[u8], count: usize) -> Result<Vec<Vec<u8>>> {
    let mut rest = input;
    let mut units = Vec::with_capacity(count);

    for _ in 0..count {
        let unit = media::read_unit(rest)?;
        units.push(unit.to_vec());
        rest = &rest[size_of::<u64>() + unit.len()..];
    }
    Ok(units)
}

/// The conditions `row`'s request carries.
fn conditions_of(row: &MediaRow) -> Result<proto::video::Conditions> {
    proto::video::Conditions::decode(&row.conditioning).map_err(|error| {
        Error::Invalid(format!(
            "request {}'s conditioning does not decode: {error}",
            row.request.0
        ))
    })
}

/// The bytes of each product a read of `conditioning` writes, as `format` lays its conditions
/// out, for the products its conditions call for, in product order.
fn extents(
    conditioning: &proto::video::Conditions,
    format: &MediaFormat,
) -> Result<Vec<(&'static str, u64)>> {
    let row = patch_row(format)?;
    let (mut pixels, mut samples, mut patches) = (0, 0, 0);

    for condition in &conditioning.conditions {
        pixels += condition.pixel_bytes() * FLOAT;
        if let Some(track) = condition.media.audio() {
            samples += 2 * conditions::track_samples(format, track.samples)? * FLOAT;
        }
        if let Some(vision) = &condition.vision {
            patches += vision.grid.patches() * row;
        }
    }

    Ok([
        (CONDITION_PIXELS, pixels),
        (CONDITION_SAMPLES, samples),
        (VISION_PIXELS, patches),
    ]
    .into_iter()
    .filter(|(_, bytes)| *bytes > 0)
    .collect())
}

/// The bytes of one patch row a vision tower of `format` reads.
fn patch_row(format: &MediaFormat) -> Result<u64> {
    let element = ops::abi::bytes(format.patches.dtype).ok_or_else(|| {
        Error::Invalid(format!(
            "a patch row of {} elements has no whole-byte size",
            ops::abi::name(format.patches.dtype)
        ))
    })?;

    Ok(u64::from(format.patches.width()) * element)
}

/// `bytes` as the `f32` elements a condition is written into. A region whose base is not aligned
/// for them is refused rather than viewed from a shifted start.
fn floats(bytes: &mut [u8]) -> Result<&mut [f32]> {
    bytemuck::try_cast_slice_mut(bytes).map_err(|error| {
        Error::Invalid(format!("a condition region cannot hold f32 rows: {error}"))
    })
}

/// The bytes one condition's media was published as, read from shared memory.
fn published_bytes(source: &proto::video::MediaLocator) -> Result<Vec<u8>> {
    use std::os::unix::fs::FileExt as _;

    let read = |error: std::io::Error| {
        Error::Invalid(format!(
            "reading the published media {:?} of {} byte(s): {error}",
            source.name, source.bytes
        ))
    };
    let file = proto::SharedMemory::open(&source.name, false).map_err(read)?;
    let length = host_bytes(source.bytes)?;
    let mut bytes = vec![0_u8; length];
    file.read_exact_at(&mut bytes, 0).map_err(read)?;
    Ok(bytes)
}

/// Decodes one request's conditions and lays them out as the model's encoders read them: the
/// bytes of each product a read writes, as [`extents`] states them.
///
/// `crates/media` decodes the bytes into RGB frames and stereo PCM at their planned extents;
/// the model's format then lays them out. A reference video's frames past its encoded prefix are
/// the vision tower's alone, which is why the reader hands them back.
fn read_conditions(
    media: &media::MediaRunner,
    (frame_rate, audio_rate): (u32, u32),
    conditioning: &proto::video::Conditions,
    format: &MediaFormat,
) -> Result<Vec<Vec<u8>>> {
    use media::read::{
        AudioTransform, ConditionProducts, ConditionSource, ImageTransform, VideoClip,
    };
    use proto::video::ConditionMedia;

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
            ConditionMedia::Video { .. } => {
                proto::SharedMemory::open(&condition.source.name, false)
                    .map(Some)
                    .map_err(|error| {
                        Error::Invalid(format!(
                            "opening the published media {:?}: {error}",
                            condition.source.name
                        ))
                    })
            }
            _ => Ok(None),
        })
        .collect::<Result<Vec<_>>>()?;

    let index = |value: u64| {
        usize::try_from(value).map_err(|_| Error::Invalid("a track is larger than memory".into()))
    };
    let audio_transform = |track: &proto::video::AudioClip| -> Result<AudioTransform> {
        Ok(AudioTransform {
            native_rate: track.sample_rate,
            start: index(track.start_sample)?,
            source_samples: index(track.source_samples)?,
            samples: index(track.samples)?,
            rate: audio_rate,
        })
    };

    let mut reads = Vec::with_capacity(conditions.len());
    let (mut pixel_rows, mut sample_rows) = (0_usize, 0_usize);
    for (index, condition) in conditions.iter().enumerate() {
        let (image, video) = match &condition.media {
            ConditionMedia::Image(fit) => (
                Some(ImageTransform {
                    resized: (fit.resized.width, fit.resized.height),
                    window: (fit.left, fit.top, fit.size.width, fit.size.height),
                }),
                None,
            ),
            ConditionMedia::Video { clip, .. } => (
                None,
                Some(VideoClip {
                    frame_rate,
                    canvas: (clip.canvas.width, clip.canvas.height),
                    frames: clip.frames,
                    vae_frames: clip.vae_frames,
                    start_frame: clip.start_frame,
                }),
            ),
            ConditionMedia::Audio(_) => (None, None),
        };
        let pixels = host_bytes(condition.pixel_bytes() / 3)?;
        let audio = condition.media.audio().map(audio_transform).transpose()?;
        let samples = audio.map_or(0, |track| track.samples);
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

    let began = Instant::now();
    let mut rgb = vec![0_u8; pixel_rows * 3];
    let mut pcm = vec![0_f32; sample_rows * 2];
    let decoded = media.read_conditions(
        &reads,
        &mut ConditionProducts {
            pixels: (pixel_rows > 0).then_some(rgb.as_mut_slice()),
            samples: (sample_rows > 0).then_some(pcm.as_mut_slice()),
        },
    )?;
    let read_ms = began.elapsed().as_millis() as u64;

    // The regions the encoders read, each as many bytes as its product's block holds of it.
    let extents = extents(conditioning, format)?;
    let region = |kind: &str| -> Result<Vec<u8>> {
        let bytes = extents
            .iter()
            .find(|(name, _)| *name == kind)
            .map_or(0, |(_, bytes)| *bytes);
        Ok(vec![0_u8; host_bytes(bytes)?])
    };
    let (mut pixels, mut samples, mut patches) = (
        region(CONDITION_PIXELS)?,
        region(CONDITION_SAMPLES)?,
        region(VISION_PIXELS)?,
    );
    let patch_row = host_bytes(patch_row(format)?)?;

    // Each condition's own rows, one condition after another.
    let (mut pixels_at, mut samples_at, mut patches_at) = (0_usize, 0_usize, 0_usize);
    for (index, condition) in conditions.iter().enumerate() {
        let read = &reads[index];
        if let Some((frames, canvas)) = condition.media.pixels() {
            let source = &rgb[read.pixels.start * 3..read.pixels.end * 3];
            let out = &mut floats(&mut pixels)?[pixels_at..pixels_at + source.len()];
            conditions::pixels(
                &format.pixels,
                source,
                frames,
                canvas.height,
                canvas.width,
                out,
            )?;
            pixels_at += source.len();
        }

        if let Some(track) = condition.media.audio() {
            let padded = 2 * host_bytes(conditions::track_samples(format, track.samples)?)?;
            conditions::track(
                format,
                &pcm[read.samples.start * 2..read.samples.end * 2],
                track.samples,
                &mut floats(&mut samples)?[samples_at..samples_at + padded],
            )?;
            samples_at += padded;
        }

        if let Some(vision) = &condition.vision {
            let grid = [vision.grid.t, vision.grid.h, vision.grid.w];
            let rows = host_bytes(vision.grid.patches())?;
            let out = &mut patches[patches_at * patch_row..(patches_at + rows) * patch_row];

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
                        Error::Invalid("a video samples more frames than a u32".into())
                    })?;
                    conditions::pack_block(
                        &format.patches,
                        &sampled,
                        frames,
                        (height, width),
                        grid,
                        out,
                    )?;
                }
                None => {
                    let (frames, canvas) = condition.media.pixels().ok_or_else(|| {
                        Error::Invalid("the conditioner reads a condition with no pixels".into())
                    })?;
                    conditions::pack_block(
                        &format.patches,
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
    tracing::info!(
        read_ms,
        total_ms = began.elapsed().as_millis() as u64,
        "the conditions were decoded and laid out"
    );

    Ok([pixels, samples, patches]
        .into_iter()
        .filter(|region| !region.is_empty())
        .collect())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::batch::{Columns, Extent, Shape};
    use crossbeam_channel::Receiver;
    use nn::DType;
    use nn::capability::{PatchFormat, PixelNorm};
    use proto::request::RequestId;
    use storage::Key;
    use storage::runtime::{self, Completion, GroupId, OwnerId, Request, Tier};

    use super::*;
    use crate::executor::Route;
    use crate::worker::{Sequences, Worker};

    /// The worker group, and the regions its ranks map: the slot rows, each of two ranks'
    /// product buffer, and the host encoder region.
    const GROUP: GroupId = GroupId(4);
    const SLOTS: u8 = 0;
    const PRODUCTS: [u8; 2] = [1, 3];
    const ENCODERS: u8 = 2;
    const UNIT_SIZES: [(u8, u64); 4] = [
        (SLOTS, 1),
        (PRODUCTS[0], UNIT_BYTES),
        (PRODUCTS[1], UNIT_BYTES),
        (ENCODERS, 512),
    ];

    /// The classes of the product kinds, and the positions of the request's video units, encoded
    /// units (one block per unit) and audio samples in its product table: those of a request of
    /// two units without conditions.
    const CLASSES: [(&str, u8); 6] = [
        (CONDITION_PIXELS, 5),
        (CONDITION_SAMPLES, 6),
        (VISION_PIXELS, 7),
        (VIDEO_UNITS, 8),
        (ENCODED_UNITS, 9),
        (AUDIO_SAMPLES, 10),
    ];
    const VIDEO_AT: u32 = 3;
    const ENCODED_AT: [u32; 2] = [6, 7];
    const AUDIO_AT: u32 = 8;

    /// The request's timeline: 39 frames in units of 17 and 22, at 16 by 16, a round a unit.
    const FRAMES: u32 = 39;
    const UNITS: [u32; 2] = [17, 22];
    const SIDE: u32 = 16;

    /// The request, the owner it is on the group, and its slot.
    const REQUEST: RequestId = RequestId(7);
    const OWNER: OwnerId = OwnerId(1);
    const SLOT: u32 = 2;

    /// A bound on the tests' waits that only a defect reaches.
    const LONG: std::time::Duration = std::time::Duration::from_secs(60);

    fn config(phases: &[Phase]) -> HostConfig {
        let half = PixelNorm {
            mean: [0.5; 3],
            std: [0.5; 3],
        };

        HostConfig {
            phases: phases.to_vec(),
            classes: CLASSES.to_vec(),
            timelines: vec![(FRAMES, UNITS.to_vec())],
            round: RoundConfig {
                ranks: vec![0],
                units: 1,
            },
            frame_rate: 24,
            audio_rate: 32_000,
            format: MediaFormat {
                pixels: half,
                patches: PatchFormat {
                    side: 2,
                    temporal: 2,
                    merge: 2,
                    channels: 3,
                    norm: half,
                    dtype: DType::BF16,
                },
                hop: 800,
            },
        }
    }

    fn kernels() -> Arc<Generation> {
        Arc::new(Generation::new(0, []))
    }

    /// The media memory of rank `rank`, which holds no device: a product buffer of `units` units
    /// and the rank's part of the host encoder region.
    fn memory(rank: usize, units: u32) -> BufferMemory {
        let index = u32::try_from(rank).unwrap();

        BufferMemory::meta(
            index,
            (PRODUCTS[rank], units),
            (SLOTS, 4),
            0,
            &[(ENCODERS, 512, 8)],
        )
        .unwrap()
    }

    /// A media host of `phases` over `memory`.
    fn host(
        phases: &[Phase],
        memory: BufferMemory,
        pool: &Arc<ThreadPool>,
        media: &Arc<media::MediaRunner>,
    ) -> MediaHost {
        MediaHost::new(
            config(phases),
            kernels(),
            Arc::new(memory),
            None,
            Arc::clone(pool),
            Arc::clone(media),
        )
        .unwrap()
    }

    fn pool() -> Arc<ThreadPool> {
        Arc::new(ThreadPool::new("media-test", 8, 1).unwrap())
    }

    fn runner() -> Arc<media::MediaRunner> {
        Arc::new(media::MediaRunner::new(media::MediaConfig::from_env()))
    }

    /// The tables of rank `rank`, holding nothing.
    fn tables(rank: usize) -> BlockTables {
        BlockTables::new(GROUP, &[SLOTS, PRODUCTS[rank], ENCODERS], &UNIT_SIZES)
    }

    /// The request's row of a host batch, naming `results` of which it reads the first `reads`.
    fn row(results: Vec<u32>, reads: u32) -> MediaRow {
        MediaRow {
            request: REQUEST,
            slot: SLOT,
            conditioning: Vec::new(),
            step: 0,
            steps: 4,
            seed: 1,
            width: SIDE,
            height: SIDE,
            frames: FRAMES,
            rounds: 0,
            unit: 0,
            rank: None,
            image: None,
            results,
            reads,
        }
    }

    /// Batch `id` of host phase `phase` over `rows`.
    fn batch(id: u64, phase: Phase, rows: Vec<MediaRow>) -> Batch {
        Batch {
            id: BatchId(id),
            group: GROUP,
            phase,
            generation: 0,
            shape: Shape {
                rows: u32::try_from(rows.len()).unwrap(),
                tokens: 0,
                variant: variant(phase).unwrap(),
                extent: Extent::None,
            },
            requests: rows.iter().map(|row| row.request).collect(),
            columns: Columns::default(),
            media: rows,
            new_slots: Vec::new(),
        }
    }

    /// The bytes of the request's video units block, of each of its encoded units blocks and of
    /// its audio samples block, in product buffer units.
    fn blocks() -> [u32; 3] {
        let frame = u64::from(SIDE * SIDE * 3);
        let samples = media::audio_samples(FRAMES as usize, 24, 32_000) as u64;

        [
            frames(&UNITS) * frame,
            MediaHost::encoded_bytes(&UNITS, SIDE, SIDE),
            samples * STEREO,
        ]
        .map(|bytes| u32::try_from(bytes.div_ceil(UNIT_BYTES)).unwrap())
    }

    /// The product buffer units the request's blocks take on a rank, one after another.
    fn held() -> u32 {
        let [video, encoded, audio] = blocks();
        video + 2 * encoded + audio
    }

    /// The completions that attach the request in `SLOT` and give it its video units, encoded
    /// units and audio samples blocks one after another from unit 0 of each of `regions`, the
    /// product buffers of the ranks of a process, with the requests they answer.
    fn admission(regions: &[u8]) -> Vec<(u32, Request, Completion)> {
        let attach = Request::Attach {
            owner: OWNER,
            group: GROUP,
        };
        let mut answers = vec![(SLOT, attach, Completion::Attached { slot: SLOT })];

        let mut first = 0;
        let [video, encoded, audio] = blocks();
        for ((kind, at), units) in [
            ((VIDEO_UNITS, VIDEO_AT), video),
            ((ENCODED_UNITS, ENCODED_AT[0]), encoded),
            ((ENCODED_UNITS, ENCODED_AT[1]), encoded),
            ((AUDIO_SAMPLES, AUDIO_AT), audio),
        ] {
            let allocate = Request::Allocate {
                owner: OWNER,
                class: CLASSES.iter().find(|(name, _)| *name == kind).unwrap().1,
                at,
                count: 1,
                units,
            };
            let allocated = Completion::Allocated {
                runs: regions
                    .iter()
                    .map(|&region| (region, first, units))
                    .collect(),
                evicted: Vec::new(),
            };
            answers.push((SLOT, allocate, allocated));
            first += units;
        }
        answers
    }

    /// Performs `batch` on `host` over `mirrors`, host mirrors of the product buffers by region,
    /// as the devices would: what each row reads is taken from them, an encoded unit's row as far
    /// as its framed unit reaches, and each output it produced is written to each of its spans.
    /// Returns what each row produced.
    fn perform(
        host: &mut MediaHost,
        tables: &BlockTables,
        mirrors: &mut HashMap<u8, Vec<u8>>,
        batch: &Batch,
    ) -> Vec<Produced> {
        let mut produced = Vec::new();

        let transfers = host.transfers(batch, tables).unwrap();
        for (index, (reads, writes)) in transfers.into_iter().enumerate() {
            let mut input = Vec::new();
            for (region, address, bytes) in reads {
                let span = &mirrors[&region][address as usize..(address + bytes) as usize];
                let framed = match batch.phase {
                    Phase::Mux => 8 + u64::from_ne_bytes(span[..8].try_into().unwrap()),
                    _ => bytes,
                };
                input.extend_from_slice(&span[..framed as usize]);
            }

            let output = host.work(batch, index, input).unwrap().run().unwrap();
            for (spans, bytes) in writes.iter().zip(&output.0) {
                for &(region, address, _) in spans {
                    let at = address as usize;
                    let mirror = mirrors.get_mut(&region).unwrap();
                    mirror[at..at + bytes.len()].copy_from_slice(bytes);
                }
            }
            produced.push(output);
        }
        produced
    }

    /// The bytes of a published artifact, claimed from shared memory.
    fn claimed(artifact: &MediaOutput) -> Vec<u8> {
        let crate::batch::ArtifactHandle::PosixShm { name } = &artifact.handle;

        // SAFETY: the media host published the object and writes it no more.
        let media =
            unsafe { crate::batch::media::SharedMedia::open(name, artifact.bytes) }.unwrap();
        media.as_bytes().to_vec()
    }

    #[test]
    fn a_mux_batch_reads_the_encoded_units_its_rows_name_and_assembles_the_artifact() {
        let (pool, media) = (pool(), runner());
        let [video, encoded, _] = blocks();
        let units = held();

        // Rank 0 of a process of two ranks, each backing the request's products.
        let mut memories = [memory(0, units), memory(1, units)];
        BufferMemory::connect(&mut memories).unwrap();
        let [own, _] = memories;
        let mut host = host(
            &[Phase::VideoEncode, Phase::AudioEncode, Phase::Mux],
            own,
            &pool,
            &media,
        );
        let mut tables = tables(0).with_peers(&PRODUCTS);
        for (slot, request, completion) in admission(&PRODUCTS) {
            tables
                .map(slot, runtime::RequestId(1), &request, &completion)
                .unwrap();
        }
        let buffer = vec![0_u8; (u64::from(units) * UNIT_BYTES) as usize];
        let mut mirrors: HashMap<u8, Vec<u8>> =
            PRODUCTS.map(|region| (region, buffer.clone())).into();

        // The decoded frames in rank 0's run: each unit's frames are one value.
        let frame = (SIDE * SIDE * 3) as usize;
        let decoded = mirrors.get_mut(&PRODUCTS[0]).unwrap();
        decoded[..UNITS[0] as usize * frame].fill(0x40);
        decoded[UNITS[0] as usize * frame..FRAMES as usize * frame].fill(0xa0);

        // Each round's encode writes its unit's encoding into the run of each rank of the unit's
        // own encoded units block, from its first byte.
        let rows = [0, 1].map(|unit| u64::from(video + unit * encoded) * UNIT_BYTES);
        for unit in 0..2 {
            let mut encode = row(vec![VIDEO_AT, ENCODED_AT[unit as usize]], 1);
            encode.unit = unit;
            perform(
                &mut host,
                &tables,
                &mut mirrors,
                &batch(1, Phase::VideoEncode, vec![encode]),
            );

            let at = rows[unit as usize] as usize;
            let length = u64::from_ne_bytes(mirrors[&PRODUCTS[0]][at..at + 8].try_into().unwrap());
            let framed = at..at + 8 + length as usize;
            assert!(length > 0, "unit {unit}'s row holds its encoding");
            assert_eq!(
                mirrors[&PRODUCTS[0]][framed.clone()],
                mirrors[&PRODUCTS[1]][framed],
                "rank 1's run holds unit {unit}'s row as well"
            );
        }

        // The silent track, then a muxing call per round, each reading its unit's block.
        perform(
            &mut host,
            &tables,
            &mut mirrors,
            &batch(2, Phase::AudioEncode, vec![row(vec![AUDIO_AT], 1)]),
        );
        for (round, at) in rows.into_iter().enumerate() {
            let mut hand = row(vec![ENCODED_AT[round]], 1);
            hand.rounds = 1;
            let handing = batch(3, Phase::Mux, vec![hand]);

            let bytes = media::encoded_unit_bytes(UNITS[round], SIDE, SIDE) as u64;
            assert_eq!(
                host.transfers(&handing, &tables).unwrap(),
                [(vec![(PRODUCTS[0], at, bytes)], Vec::new())],
                "round {round} reads its unit's row"
            );
            let produced = perform(&mut host, &tables, &mut mirrors, &handing);
            assert_eq!(produced, [(Vec::new(), None)]);
        }

        // The final call reads nothing and answers with the request's MP4.
        let last = batch(4, Phase::Mux, vec![row(Vec::new(), 0)]);
        assert_eq!(
            host.transfers(&last, &tables).unwrap(),
            [(Vec::new(), Vec::new())]
        );
        let produced = perform(&mut host, &tables, &mut mirrors, &last);
        let artifact = claimed(produced[0].1.as_ref().unwrap());

        assert_eq!(
            &artifact[4..8],
            b"ftyp",
            "an MP4 begins with its file type box"
        );
        for kind in [b"moov", b"mdat"] {
            assert!(artifact.windows(4).any(|box_type| box_type == kind));
        }

        // A round past the timeline's units is refused by name.
        let mut past = row(vec![VIDEO_AT, ENCODED_AT[0]], 1);
        past.unit = 2;
        let refused = host.transfers(&batch(5, Phase::VideoEncode, vec![past]), &tables);
        assert!(refused.unwrap_err().to_string().contains("2 units"));
    }

    #[test]
    fn a_video_encode_reads_each_unit_from_the_run_of_the_rank_that_decoded_it() {
        let (pool, media) = (pool(), runner());

        // The timeline's two units are one round, dealt to ranks 0 and 1 of the process, each
        // of which decoded its unit into its own run of the video units block.
        let mut memories = [0, 1].map(|rank| memory(rank, held()));
        BufferMemory::connect(&mut memories).unwrap();
        let [own, _] = memories;
        let mut config = config(&[Phase::VideoEncode]);
        config.round = RoundConfig {
            ranks: vec![0, 1],
            units: 1,
        };
        let host = MediaHost::new(config, kernels(), Arc::new(own), None, pool, media).unwrap();
        let mut tables = tables(0).with_peers(&PRODUCTS);
        for (slot, request, completion) in admission(&PRODUCTS) {
            tables
                .map(slot, runtime::RequestId(1), &request, &completion)
                .unwrap();
        }

        // Unit 0's frames from rank 0's run, then unit 1's from rank 1's, after unit 0's frames.
        let encode = batch(
            1,
            Phase::VideoEncode,
            vec![row(vec![VIDEO_AT, ENCODED_AT[0], ENCODED_AT[1]], 1)],
        );
        let transfers = host.transfers(&encode, &tables).unwrap();
        let frame = u64::from(SIDE * SIDE * 3);
        let (first, second) = (u64::from(UNITS[0]) * frame, u64::from(UNITS[1]) * frame);
        assert_eq!(
            transfers[0].0,
            [(PRODUCTS[0], 0, first), (PRODUCTS[1], first, second)]
        );
    }

    #[test]
    fn a_video_encode_in_each_process_encodes_the_units_its_ranks_decoded_into_their_blocks() {
        let (pool, media) = (pool(), runner());
        let [video, encoded, _] = blocks();
        let units = held();

        // The round's two units are dealt to rank 0, of one process, and rank 1, of another:
        // each process maps the runs of its own rank alone.
        let mut config = config(&[Phase::VideoEncode]);
        config.round = RoundConfig {
            ranks: vec![0, 1],
            units: 1,
        };
        let encode = batch(
            1,
            Phase::VideoEncode,
            vec![row(vec![VIDEO_AT, ENCODED_AT[0], ENCODED_AT[1]], 1)],
        );
        let buffer = vec![0_u8; (u64::from(units) * UNIT_BYTES) as usize];
        let frame = u64::from(SIDE * SIDE * 3);
        let rows = [0, 1].map(|unit| (u64::from(video + unit * encoded) * UNIT_BYTES) as usize);

        for rank in 0..2 {
            let memory = Arc::new(memory(rank, units));
            let (pool, media) = (Arc::clone(&pool), Arc::clone(&media));
            let mut host =
                MediaHost::new(config.clone(), kernels(), memory, None, pool, media).unwrap();
            let mut tables = tables(rank);
            for (slot, request, completion) in admission(&PRODUCTS[rank..=rank]) {
                tables
                    .map(slot, runtime::RequestId(1), &request, &completion)
                    .unwrap();
            }

            // The process's rank decoded its unit into its own run, after the earlier units'
            // frames.
            let mut mirrors: HashMap<u8, Vec<u8>> = [(PRODUCTS[rank], buffer.clone())].into();
            let before = frames(&UNITS[..rank]) * frame;
            let bytes = u64::from(UNITS[rank]) * frame;
            let decoded = mirrors.get_mut(&PRODUCTS[rank]).unwrap();
            decoded[before as usize..(before + bytes) as usize].fill(0x40);

            // It reads that unit's frames alone, and writes that unit's block alone.
            let transfers = host.transfers(&encode, &tables).unwrap();
            assert_eq!(transfers[0].0, [(PRODUCTS[rank], before, bytes)]);
            let produced = perform(&mut host, &tables, &mut mirrors, &encode);
            assert_eq!(
                produced[0].0.len(),
                1,
                "rank {rank}'s process encodes one unit"
            );

            let mirror = &mirrors[&PRODUCTS[rank]];
            let length = |at: usize| u64::from_ne_bytes(mirror[at..at + 8].try_into().unwrap());
            assert!(
                length(rows[rank]) > 0,
                "unit {rank}'s block holds its encoding"
            );
            assert_eq!(
                length(rows[1 - rank]),
                0,
                "unit {} is the other process's to encode",
                1 - rank
            );
        }
    }

    /// A runner of the device phase `Decode`, which answers each request of a batch.
    struct Device {
        graphs: Graphs,
    }

    impl Runner for Device {
        fn phases(&self) -> &[Phase] {
            &[Phase::Decode]
        }

        fn buckets(&self) -> Vec<Bucket> {
            Vec::new()
        }

        fn forward(&self, bucket: &Bucket) -> Result<Vec<Tensor>> {
            Err(Error::Unsupported(format!("no forward of {bucket}")))
        }

        fn idle_inputs(&self, _: &Bucket, _: &mut [u8]) -> Result<()> {
            Ok(())
        }

        fn prepare_inputs(&mut self, batch: &Batch, _: &BlockTables) -> Result<Bucket> {
            Err(Error::Unsupported(format!("no inputs of {}", batch.id.0)))
        }

        fn execute(&mut self, batch: &Batch, _: &BlockTables) -> Result<BatchOutput> {
            Ok(BatchOutput {
                calls: batch.requests.iter().map(|_| performed()).collect(),
                ..BatchOutput::default()
            })
        }

        fn graphs(&self) -> &Graphs {
            &self.graphs
        }
    }

    /// Host work that waits until its sender is dropped: what holds a pool's thread.
    struct Gate(Receiver<()>);

    impl HostAction for Gate {
        type Output = ();

        fn run(&self) -> Result<()> {
            let _ = self.0.recv();
            Ok(())
        }

        fn release(&self) -> Result<()> {
            Ok(())
        }

        fn input_status(&self) -> Outcome {
            Outcome::Success(())
        }
    }

    /// A route over one worker per runner set given, rank `i` of `GROUP` over `runners[i]`,
    /// holding the request's admission.
    fn route(runners: Vec<Vec<Box<dyn Runner>>>) -> Route {
        let ranks = runners.len();
        let sequences = Sequences::new(GROUP, ranks, Vec::new());
        let workers: Vec<Arc<Worker>> = runners
            .into_iter()
            .enumerate()
            .map(|(rank, runners)| {
                let worker = Worker::spawn(
                    None,
                    runners,
                    kernels(),
                    tables(rank),
                    Arc::clone(&sequences),
                    rank,
                    crate::worker::COPY_BUDGET,
                );
                Arc::new(worker.unwrap())
            })
            .collect();

        let route = Route::new(vec![sequences], workers, Vec::new()).unwrap();
        for (slot, request, completion) in admission(&PRODUCTS[..ranks]) {
            returned(
                route
                    .map(GROUP, slot, runtime::RequestId(1), request, completion)
                    .unwrap(),
            );
        }
        route
    }

    /// What `submission`, which succeeds, returned.
    fn returned(submission: crate::worker::completion::Submission) -> (BatchOutput, Vec<Request>) {
        match submission.wait_timeout(LONG) {
            Some(Outcome::Success(returned)) => returned,
            Some(Outcome::Failed(error)) => panic!("the job failed: {error}"),
            other => panic!("the job did not succeed: {other:?}"),
        }
    }

    /// Device batch `id` of one request.
    fn device(id: u64) -> Batch {
        Batch {
            id: BatchId(id),
            group: GROUP,
            phase: Phase::Decode,
            generation: 0,
            shape: Shape::decode(1, Variant::DecodeGreedy),
            requests: vec![RequestId(100 + id)],
            columns: Columns::default(),
            media: Vec::new(),
            new_slots: Vec::new(),
        }
    }

    #[test]
    fn a_host_batch_takes_its_number_in_order_and_holds_it_while_device_batches_run() {
        let (pool, media) = (pool(), runner());
        let units = held();
        let device_runner = Device {
            graphs: Graphs::meta(kernels(), 0),
        };
        let host = host(&[Phase::AudioEncode], memory(0, units), &pool, &media);
        let route = route(vec![vec![Box::new(device_runner), Box::new(host)]]);

        // The pool's one thread is held, so the host batch's work waits behind it.
        let (release, gate) = crossbeam_channel::bounded::<()>(0);
        let _held = pool.submit(Gate(gate)).unwrap();

        returned(route.submit(device(1)).unwrap());
        let encode = batch(2, Phase::AudioEncode, vec![row(vec![AUDIO_AT], 1)]);
        let host_batch = route.submit(encode).unwrap();
        returned(route.submit(device(3)).unwrap());

        // Batch 3 ran and ended while the host batch's work waited: the group's completed word
        // stays below the host batch, and a release stamped now carries the last number launched.
        assert!(host_batch.outcome().is_none());
        assert_eq!(route.completed(GROUP).unwrap(), 1);
        let release_block = |after| Request::Release {
            owner: OWNER,
            class: 10,
            at: AUDIO_AT,
            count: 1,
            after,
        };
        let (_, stamped) = returned(route.stamp(GROUP, vec![release_block(Vec::new())]).unwrap());
        assert_eq!(stamped, [release_block(vec![(GROUP, 3)])]);

        // The work runs once the pool's thread is free, and the batch ends with it: the track is
        // encoded from the block the first turn read, though its entry has left the tables.
        drop(release);
        let (output, finished) = returned(host_batch);
        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].status, CallStatus::Ok);
        assert!(finished.is_empty());
        assert_eq!(route.completed(GROUP).unwrap(), 3);
    }

    #[test]
    fn a_host_batch_with_no_rows_runs_the_copies_queued_for_its_group() {
        let (pool, media) = (pool(), runner());
        let host = host(&[Phase::Mux], memory(0, 16), &pool, &media);
        let route = route(vec![vec![Box::new(host)]]);

        // A store of an encoder output of two product units to a host block of one 512-byte unit.
        let key = Key([4; 32]);
        let store = Request::Store {
            group: GROUP,
            keys: vec![key],
            tier: Tier::Host,
        };
        let copy = Completion::Copy {
            group: GROUP,
            keys: vec![key],
            from: vec![(PRODUCTS[0], 4, 2)],
            to: vec![(ENCODERS, 1, 1)],
            evicted: Vec::new(),
            encodings: None,
            parts: Vec::new(),
        };
        returned(
            route
                .map(GROUP, 0, runtime::RequestId(10), store, copy)
                .unwrap(),
        );

        // The batch takes number 1 and ends the copy that started with it, answering no call.
        let (output, finished) = returned(route.submit(batch(1, Phase::Mux, Vec::new())).unwrap());
        assert!(output.calls.is_empty());
        assert_eq!(
            finished,
            [Request::Finish {
                copy: runtime::RequestId(10),
                ok: true,
                after: vec![(GROUP, 1)],
            }]
        );
    }

    #[test]
    fn a_host_batch_whose_rows_name_no_rank_runs_once_on_the_first_rank_serving_its_phase() {
        let pool = pool();
        let units = held();
        let assemblies = [runner(), runner()];
        let runners = (0..2)
            .map(|rank| {
                let host = host(
                    &[Phase::AudioEncode],
                    memory(rank, units),
                    &pool,
                    &assemblies[rank],
                );
                vec![Box::new(host) as Box<dyn Runner>]
            })
            .collect();
        let route = route(runners);

        let encode = batch(1, Phase::AudioEncode, vec![row(vec![AUDIO_AT], 1)]);
        let (output, _) = returned(route.submit(encode).unwrap());
        assert_eq!(output.calls[0].status, CallStatus::Ok);

        // Rank 0 holds the request's assembly with its track; rank 1 encoded nothing.
        let twice = assemblies[0].set_audio(REQUEST.0, vec![1]).unwrap_err();
        assert!(twice.to_string().contains("twice"), "{twice}");
        let none = assemblies[1].set_audio(REQUEST.0, vec![1]).unwrap_err();
        assert!(none.to_string().contains("no media assembly"), "{none}");
    }
}

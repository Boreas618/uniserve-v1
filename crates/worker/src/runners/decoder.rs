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
//! splice tail the row writes. An audio decode is the request's track, one unit. Each unit is
//! one replay of its bucket's graph on the runner's stream, in unit order:
//!
//! 1. the unit's latents are copied into the runner's latent buffer: a clip's window of the
//!    trajectory, the latent frames from `Decoder::clip_start`, or the whole track; a clip whose
//!    splice fades against its predecessor has the predecessor's tail copied into the runner's
//!    tail buffer;
//! 2. the graph replays: the window gathered and decoded into a raster, which is placed and packed
//!    into bytes under every distinct splice of the deployment's timelines; or the track decoded
//!    and interleaved into 16-bit PCM;
//! 3. the unit's output is copied into its product: the bytes of the unit's own splice into the
//!    video units block from the splice's first frame, or the PCM into the audio samples block;
//!    a clip that a successor fades against copies the raster's last frames into its tail block.
//!
//! Every address is computed and every block checked before anything is issued: a position the
//! tables hold no entry for, a write to a read-only entry, or bytes past a block refuse the
//! batch. The copies are the driver's, so the graph's launches are the only kernels of a call.
//!
//! ## Why a clip's graph places every distinct splice
//!
//! A captured graph's launches follow from its bucket, and a splice's runs, their number and
//! their frames, follow from the clip's place in its timeline. A canvas's graph therefore places
//! the decoded clip under each distinct splice of the deployment's timelines (`first` taken as
//! zero), and a unit keeps the bytes of its own. That costs the packing of a few clips' frames
//! beside the decode, and keeps one graph per canvas.
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
//! ## The tail between rounds
//!
//! Within a round the units run on one rank in order, and a unit reads the tail the unit before
//! it wrote on the same stream. A round after the first reads the tail of the round before it,
//! which the scheduler gives it only once that round completed, and `execute` returns only once
//! its stream passed every copy: the tail's write is ordered before its read by completion, and
//! no event passes between executors. A round reads the tail from the run of the rank it runs
//! on: a round's last tail is written into this rank's run of the tail block, and lies in
//! another rank's run only once copied there.

use std::sync::Arc;
use std::time::Instant;

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
use crate::runners::{address, performed};
use crate::{Error, Result};

/// The channels of a decoded raster and of its packed bytes: red, green and blue.
const CHANNELS: u64 = rgb::CHANNELS as u64;

/// The channels of the interleaved PCM.
const PCM_CHANNELS: u32 = 2;

/// What one unit of a call moves around its replay: the copies into the runner's buffers before
/// it; the forward's output the unit keeps, with the address in the product buffer it lands at;
/// and the tail it leaves its successor, with the address it lands at and its frames.
type Unit = (PitchedBatch, (usize, u64), Option<(u64, u32)>);

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
    /// The distinct splices of the deployment's timelines, each from frame zero, in the order a
    /// clip bucket's forward returns their bytes after its raster.
    splices: Vec<Splice>,
    /// The latent buffer from byte 0 and the tail buffer from byte `tail`, `bytes` in all;
    /// `None` for graphs that hold no device.
    buffers: Option<Arc<Storage>>,
    tail: u64,
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
    /// [`Error::Invalid`] for no bucket, a bucket that is neither a clip nor a track, a video
    /// decoder serving no timeline, a splicer of another element type, a track that is not
    /// stereo, a stream without a device or the reverse; what the model or the splicer refuses:
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

        // The distinct splices of every timeline a canvas's graph places.
        let mut splices = Vec::new();
        if let Some(splicer) = &splicer {
            for &(frames, _) in &config.timelines {
                for splice in splicer.splices(frames)? {
                    let splice = Splice { first: 0, ..splice };
                    if !splices.contains(&splice) {
                        splices.push(splice);
                    }
                }
            }
        }
        let overlap = splices.iter().map(|splice| splice.overlap).max();

        // Each bucket's layout, and the bytes its latents and the tail a clip fades against take.
        let latent = element(model.latent_type())?;
        let raster = element(model.element_type())?;
        let (mut layouts, mut latents, mut tails) = (Vec::new(), 0, 0);
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
                    let pixels = u64::from(height) * u64::from(width);
                    let video = layout.video;
                    latents = latents.max(u64::from(video.channels) * voxels(&video) * latent);
                    tails = tails.max(CHANNELS * u64::from(overlap.unwrap_or(0)) * pixels * raster);
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
        let bytes = tail + tails;

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
            splices,
            buffers,
            tail,
            bytes,
        })
    }

    /// The bytes of device memory the runner holds beside its graphs' region: its latent and
    /// tail buffers, sized for its largest bucket.
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
        if offset + shape.elem_count() as u64 * element(dtype)? > self.bytes {
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

        // A latent frame's bytes of one channel, and a decoded frame's in the raster and packed.
        let video = self.layout(bucket)?.video;
        let plane = u64::from(video.height) * u64::from(video.width);
        let plane = plane * element(self.model.latent_type())?;
        let (window, pitch) = (
            u64::from(video.frames) * plane,
            u64::from(trajectory) * plane,
        );
        let pixels = u64::from(height) * u64::from(width);
        let raster = pixels * element(self.model.element_type())?;
        let base = self.buffers.as_ref().map_or(0, |storage| storage.address());

        let mut units = Vec::with_capacity(tails.len());
        for (index, unit) in (row.unit..).take(tails.len()).enumerate() {
            let mut before = PitchedBatch::new(SourceOrder::Stream);

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
                units.push((before, (0, kept), None));
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

            // The tail it fades against: the round's own before it, or the one before the round.
            if splice.overlap > 0 {
                let previous = match index {
                    0 => row.results.get(1),
                    _ => tails.get(index - 1),
                };
                let previous = previous.ok_or_else(|| {
                    refused(format!("of unit {unit} fades and names no tail before it"))
                })?;
                let bytes = CHANNELS * u64::from(splice.overlap) * raster;
                let source = self.place(tables, row, (tail_class, *previous), false, (0, bytes))?;
                before.push(pitched(
                    (base + self.tail, bytes),
                    (source, bytes),
                    bytes,
                    1,
                )?)?;
            }

            // The bytes of its own splice, from the splice's first frame of the timeline.
            let output = self
                .splices
                .iter()
                .position(|held| *held == Splice { first: 0, ..splice })
                .ok_or_else(|| Error::Invariant(format!("{splice:?} is none the graph places")))?;
            let at = (
                u64::from(splice.first) * pixels * CHANNELS,
                u64::from(splice.keep) * pixels * CHANNELS,
            );
            let kept = self.place(tables, row, (media_class, media), true, at)?;

            let tail = match successor {
                0 => None,
                frames => {
                    let bytes = (0, CHANNELS * u64::from(frames) * raster);
                    let at = self.place(tables, row, (tail_class, tails[index]), true, bytes)?;
                    Some((at, frames))
                }
            };
            units.push((before, (1 + output, kept), tail));
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

        let bytes = elements(&track) * element(self.model.latent_type())?;
        let source = self.place(tables, row, (latents_class, latents), false, (0, bytes))?;
        let (window, channels) = self.model.pcm_window(&track)?;
        let pcm = (0, window * u64::from(channels) * 2);
        let kept = self.place(tables, row, (media_class, samples), true, pcm)?;

        let base = self.buffers.as_ref().map_or(0, |storage| storage.address());
        let mut before = PitchedBatch::new(SourceOrder::Stream);
        before.push(pitched((base, bytes), (source, bytes), bytes, 1)?)?;
        Ok((before, (0, kept), None))
    }

    /// Serves `batch`: every unit's copies and replay, on the stream. Returns when the units
    /// were issued, for a batch that names a call.
    fn serve(&self, batch: &Batch, tables: &BlockTables) -> Result<Option<Instant>> {
        // A batch that names no call carries copies alone.
        if batch.requests.is_empty() {
            return Ok(None);
        }
        let (bucket, units) = self.units(batch, tables)?;
        let stream = self.stream.as_ref().ok_or(Error::State(
            "a decoder runner with no device decodes nothing",
        ))?;

        let launched = Instant::now();
        for (before, kept, tail) in &units {
            self.graphs.prepare_inputs(self, &bucket, |_| Ok(()))?;

            // SAFETY: every range lies in a block the tables give this call or in the runner's
            // buffers (`units`), which stay mapped until the batch ends; the stream orders the
            // copies after the replay before them, whose graph read the buffers.
            unsafe { before.submit(stream) }?;
            let outputs = self.graphs.replay(&bucket)?;

            // SAFETY: as above, and the graph's outputs stay until its next replay, which the
            // stream orders after these copies.
            unsafe { after(&outputs, *kept, *tail)?.submit(stream) }?;
        }
        Ok(Some(launched))
    }

    /// The clip forward of `layout` decoding to rasters of `extent`, `(frames, rows, columns)`.
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

        let (rgb, height, width) = (CHANNELS as usize, height as usize, width as usize);
        let raster = backend::empty((rgb, frames as usize, height, width), dtype)?;
        self.model.decode(layout, &block, Some(&raster), None)?;
        let Some(splicer) = &self.splicer else {
            return Ok(vec![raster]);
        };

        // The de-normalization's constants: the three means, then the three deviations.
        let norm = splicer.pixel_norm();
        let constants: Vec<f32> = norm.mean.iter().chain(&norm.std).copied().collect();
        let constants =
            Tensor::from_slice(&constants, 2 * rgb, &Device::Cpu)?.to_device(raster.device())?;
        let (mean, std) = (constants.narrow(0, 0, rgb)?, constants.narrow(0, rgb, rgb)?);

        let mut outputs = vec![raster];
        for splice in &self.splices {
            let previous = match splice.overlap as usize {
                0 => None,
                overlap => Some(self.buffer(self.tail, (rgb, overlap, height, width), dtype)?),
            };
            let bytes = backend::empty((splice.keep as usize, height, width, rgb), DType::U8)?;

            for (run, frame) in splicer.splice(previous.as_ref(), &outputs[0], *splice)? {
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

    /// A clip bucket's forward returns its raster and then, with a splicer, the bytes of each
    /// distinct splice in turn; a track bucket's returns its PCM.
    fn forward(&self, bucket: &Bucket) -> Result<Vec<Tensor>> {
        let layout = self.layout(bucket)?;

        match bucket.extent {
            Extent::Clip {
                frames,
                height,
                width,
            } => self.decode_clip(&layout, (frames, height, width)),
            _ => self.decode_track(&layout),
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

/// The bytes of one element of `dtype`.
fn element(dtype: DType) -> Result<u64> {
    match dtype.size_in_bytes() {
        0 => Err(Error::Invalid(format!(
            "{dtype:?} has no whole-byte element"
        ))),
        bytes => Ok(bytes as u64),
    }
}

/// The elements of a latent shape per channel: its frames, rows and columns.
fn voxels(shape: &nn::LatentShape) -> u64 {
    u64::from(shape.frames) * u64::from(shape.height) * u64::from(shape.width)
}

/// The elements of a latent shape.
fn elements(shape: &nn::LatentShape) -> u64 {
    u64::from(shape.channels) * voxels(shape)
}

/// The copies of a unit's output after its replay, from the forward's `outputs`: the output it
/// keeps into the product, and the last frames of the raster into its tail block.
fn after(
    outputs: &[Tensor],
    (output, at): (usize, u64),
    tail: Option<(u64, u32)>,
) -> Result<PitchedBatch> {
    let missing = || Error::Invariant(format!("the forward returned no output {output}"));
    let mut copies = PitchedBatch::new(SourceOrder::Stream);

    let kept = outputs.get(output).ok_or_else(missing)?;
    let bytes = kept.elem_count() as u64 * element(kept.dtype())?;
    copies.push(pitched((at, bytes), (address(kept)?, bytes), bytes, 1)?)?;

    // The raster is `[3, frames, rows, columns]`: the tail is the end of each channel's frames.
    if let Some((at, frames)) = tail {
        let raster = outputs.first().ok_or_else(missing)?;
        let &[channels, length, height, width] = raster.dims() else {
            return Err(Error::Invariant(
                "a clip's raster is not four-dimensional".into(),
            ));
        };
        let plane = (height * width) as u64 * element(raster.dtype())?;
        let (kept, length) = (u64::from(frames) * plane, length as u64 * plane);

        let source = address(raster)? + length - kept;
        copies.push(pitched(
            (at, kept),
            (source, length),
            kept,
            channels as u32,
        )?)?;
    }
    Ok(copies)
}

/// The copy of `rows` rows of `width` bytes, a row the destination's pitch after the one before
/// it in the destination and the source's pitch in the source.
fn pitched(
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

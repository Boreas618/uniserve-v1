//! Video decoding: a rank's decoder of one clip output ([`VideoDecoder`]), its unit
//! forward captured once per unit shape, and the decoding of the rank's media
//! unit of a video round ([`decode_video`]).

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use tvm_ffi::Any;
use uniserve_core::Canvas;
use uniserve_cuda::{CopyBatch, DeviceId, SourceOrder, Stream};
use uniserve_ipc::Batch as BatchPlan;

use super::DECODER;
use crate::cuda_graph::{GraphMode, GraphPool, GraphTable, Lookup};
use crate::executor::products::{OutputShape, Placement, bind_outputs};
use crate::executor::video::MediaBatch;
use crate::executor::video::{Parts, decode_range, publish_bound, stream_of};
use crate::ffi::{DType, Function, Handle, Interpreter};
use crate::lane::{Capture, Completion, Job};
use crate::parallel::joins::GraphFunctions;
use crate::runner::methods::{Argument, MethodCall, Methods, RunnerClass};
use crate::runner::{RunnerSetup, device_buffer};
use crate::storage::memory::StorageView;
use crate::storage::{RequestSlots, SlotField};
use crate::stream::{Fill, Issue};
use crate::{Error, Result};

/// A clip output: the native frames and raster one unit's clip decodes to.
pub(crate) type ClipOutput = (u32, Canvas);

/// What fixes a video decoding forward's computation: the unit's clip output,
/// whether the unit is its video's first and last, and how many ring
/// members hold a unit of its round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct UnitShape {
    pub(crate) clip_output: ClipOutput,
    pub(crate) first: bool,
    pub(crate) last: bool,
    pub(crate) count: u32,
}

/// The largest extents of a video decoding forward's tensors.
pub(crate) struct DecodeCapacity {
    /// `[rows, width]` of the longest complete packed latent.
    pub(crate) latent: [u64; 2],
    /// `[frames, height, width]` bounding every unit's RGB frames.
    pub(crate) frames: [u64; 3],
    /// The largest canvas's overlap as `(elements, dtype)`.
    pub(crate) overlap: (u64, DType),
}

/// A video decoder and its RGB post-processor on one rank.
///
/// A forward reads and writes four tensors at fixed addresses, so one
/// captured graph per [`UnitShape`] serves every request: the request's
/// packed latent, the rows of it the unit's clip input reads, the request's
/// retained overlap and the unit's RGB frames. The runner copies a
/// request's values in and out of them natively around the forward.
pub(super) struct VideoDecoder {
    class: RunnerClass,
    loaded: Handle,
    pub(super) stream: Stream,
    pub(super) device: DeviceId,
    /// The runner of each clip output, in preparation order.
    runners: Vec<(ClipOutput, Methods)>,
    /// The retained overlap of each request slot (`[slots + 1, elements]`):
    /// the decoded tail a request's previous round left for its next.
    overlap: RequestSlots,
    /// `[rows, width]` FP32: a request's complete packed latent.
    latent: StorageView,
    /// `[rows]` int64: the latent rows a unit's clip input reads.
    rows: StorageView,
    /// The forward's view of the request's retained overlap.
    retained: StorageView,
    /// Flat uint8: the unit's `[frames, height, width, 3]` RGB frames.
    frames: StorageView,
    /// The clip rows of each `(frames, canvas, unit)`, as the model
    /// names them.
    unit_rows: HashMap<(u32, Canvas, u32), Vec<u8>>,
    clip_rows: Function,
    pub(super) graphs: GraphTable<UnitShape>,
    /// Captures a forward between its begin and end.
    capture: GraphFunctions,
    /// The graph pool captures allocate in; `None` serves forwards eagerly.
    pool: Option<Arc<GraphPool>>,
}

impl VideoDecoder {
    pub(crate) fn new(
        setup: RunnerSetup<'_>,
        capacity: DecodeCapacity,
        slots: u32,
        pool: Option<Arc<GraphPool>>,
    ) -> Result<Self> {
        let (elements, dtype) = capacity.overlap;
        let overlap = RequestSlots::allocate(
            &[SlotField {
                name: "video_overlap".into(),
                dtype,
                shape: vec![elements as i64],
                host: false,
            }],
            slots,
            Some(setup.device),
        )?;
        let [rows, width] = capacity.latent.map(|extent| extent as i64);
        let [frames, height, canvas_width] = capacity.frames.map(|extent| extent as i64);
        let capture = GraphFunctions::import()?;
        Ok(Self {
            device: setup.device,
            runners: Vec::new(),
            overlap,
            latent: device_buffer(setup.device, DType::F32, &[rows, width])?,
            rows: device_buffer(setup.device, DType::I64, &[rows])?,
            retained: device_buffer(setup.device, dtype, &[elements as i64])?,
            frames: device_buffer(
                setup.device,
                DType::U8,
                &[frames * height * canvas_width * 3],
            )?,
            unit_rows: HashMap::new(),
            clip_rows: Interpreter::import(&format!("{DECODER}.clip_rows"))?,
            graphs: GraphTable::new(format!("{}.decoder", setup.path), capture.release.clone()),
            capture,
            pool,
            loaded: setup.loaded.clone(),
            class: RunnerClass::new(DECODER, "decode", setup.arguments())?,
            stream: setup.stream.clone(),
        })
    }

    /// Construct the runner of every clip output, the largest first.
    ///
    /// Startup alone constructs (`DecoderRunners::prepare_startup`), for
    /// every clip output an admitted video decodes to: a serving rank
    /// constructs no decoder runner, and a decode of a clip output without one
    /// is refused.
    pub(crate) fn prepare(&mut self, clip_outputs: &[ClipOutput]) -> Result<()> {
        for &clip_output in clip_outputs {
            if self
                .runners
                .iter()
                .any(|(prepared, _)| *prepared == clip_output)
            {
                continue;
            }
            let (frames, canvas) = clip_output;
            let runner = self.class.construct(vec![Argument::Integers(vec![
                i64::from(frames),
                i64::from(canvas.width),
                i64::from(canvas.height),
            ])])?;
            self.runners.push((clip_output, runner));
        }
        Ok(())
    }

    /// The arguments of `DecoderRunner.decode` for a unit of `shape` whose
    /// clip input reads `rows` latent rows and whose video's longest unit has
    /// `longest` frames.
    fn arguments(&self, shape: UnitShape, rows: u64, longest: u32) -> Result<Vec<Argument>> {
        let canvas = shape.clip_output.1;
        let runner = self
            .runners
            .iter()
            .find(|(prepared, _)| *prepared == shape.clip_output)
            .map(|(_, runner)| runner)
            .ok_or_else(|| Error::Invariant("a video clip output has no decoder runner".into()))?;
        let frames = self.unit_frames(canvas, longest)?;
        Ok(vec![
            Argument::Handle(runner.runner().clone()),
            Argument::Tensor(self.latent.clone()),
            Argument::Tensor(self.rows.prefix(rows as i64)?),
            Argument::Tensor(self.retained.clone()),
            Argument::Tensor(frames),
            Argument::Bool(shape.first),
            Argument::Bool(shape.last),
            Argument::Integer(i64::from(shape.count)),
        ])
    }

    /// The forward's `[frames, height, width, 3]` view of the RGB frames.
    fn unit_frames(&self, canvas: Canvas, frames: u32) -> Result<StorageView> {
        let shape = [
            i64::from(frames),
            i64::from(canvas.height),
            i64::from(canvas.width),
            3,
        ];
        self.frames
            .prefix(shape.iter().product())?
            .reshape(DType::U8, &shape)
    }

    /// Read from the model the clip rows of unit `unit` of a
    /// `frames`-frame video at `canvas`. Startup reads the clip rows of every
    /// unit of every admitted video; a decode takes its rows from that
    /// table ([`Self::rows_of`]).
    pub(super) fn read_clip_rows(&mut self, frames: u32, canvas: Canvas, unit: u32) -> Result<()> {
        let key = (frames, canvas, unit);
        if !self.unit_rows.contains_key(&key) {
            let value = self.clip_rows.call_values(&[
                self.loaded.to_any(),
                Any::from(i64::from(frames)),
                Any::from(i64::from(canvas.width)),
                Any::from(i64::from(canvas.height)),
                Any::from(i64::from(unit)),
            ])?;
            let rows = crate::ffi::integers(value)
                .map_err(|error| Error::numerical(self.clip_rows.name(), &error))?;
            self.unit_rows
                .insert(key, rows.iter().flat_map(|row| row.to_ne_bytes()).collect());
        }
        Ok(())
    }

    /// The clip rows of unit `unit` of a `frames`-frame video at
    /// `canvas`, as native int64 bytes.
    fn rows_of(&self, frames: u32, canvas: Canvas, unit: u32) -> Result<&[u8]> {
        self.unit_rows
            .get(&(frames, canvas, unit))
            .map(Vec::as_slice)
            .ok_or_else(|| Error::Invalid("the rank admits no video with this media unit".into()))
    }

    /// Record into `job` the decode and reconstruction of a zero latent as
    /// a unit of `shape` whose clip input reads `rows` rows, the forward as
    /// serving records it, so kernels compile and the ring's exchange of
    /// the round shape is prepared, then the wait for the stream. Every
    /// member holding a unit of the round records it, in the same order.
    /// The forward's tensors are zero: the latent, the clip rows and the
    /// retained overlap.
    pub(crate) fn warm(
        &mut self,
        shape: UnitShape,
        rows: u64,
        longest: u32,
        job: &Job,
    ) -> Result<()> {
        let stream = self.stream.clone();
        let issue = job.on(Some(&stream));
        // SAFETY: the decoder's own inputs, kept until it closes after its
        // stream drained; only its forwards read them, recorded behind the
        // fills on the same stream.
        unsafe {
            self.latent.zero(issue)?;
            self.rows.zero(issue)?;
            self.retained.zero(issue)?;
        }
        // No shape is captured before every shape is warmed: the forward
        // is the eager call.
        self.forward(shape, rows, longest, false, job)?;
        issue.synchronize()
    }

    /// Record into `job` the capture of the forward of a warmed `shape`;
    /// whether it records one: a decoder without a graph pool serves every
    /// shape eagerly, and a captured shape is not captured again.
    pub(crate) fn capture(
        &mut self,
        shape: UnitShape,
        rows: u64,
        longest: u32,
        job: &Job,
    ) -> Result<bool> {
        if self.pool.is_none() || self.graphs.contains(&shape, None) {
            return Ok(false);
        }
        Ok(self.forward(shape, rows, longest, true, job)? == GraphMode::Capture)
    }

    /// Store the graph the job of [`Self::capture`] captured for `shape`.
    pub(crate) fn complete_capture(
        &mut self,
        shape: UnitShape,
        completion: &mut Completion,
    ) -> Result<()> {
        self.graphs.insert(shape, None, completion.captured()?)?;
        Ok(())
    }

    /// Record in `job` the forward of a unit of `shape` whose clip input reads
    /// `rows` latent rows, on the decoder's stream: the launch of the
    /// shape's captured graph, or the eager call of `DecoderRunner.decode`,
    /// which serves a shape startup did not capture. With `capture`, a
    /// shape not captured yet is captured instead of called (startup's
    /// capture of a warmed shape): the job's completion carries the graph
    /// ([`Self::complete_capture`]). A capture is counted when its graph is
    /// stored.
    fn forward(
        &mut self,
        shape: UnitShape,
        rows: u64,
        longest: u32,
        capture: bool,
        job: &Job,
    ) -> Result<GraphMode> {
        let method = |decoder: &Self| -> Result<MethodCall> {
            Ok(MethodCall::new(
                &decoder.class.forward,
                decoder.arguments(shape, rows, longest)?,
            ))
        };
        let mode = match self.graphs.lookup(&shape, None) {
            Ok(Lookup::Replay(graph)) => {
                job.on(Some(&self.stream)).launch(graph)?;
                GraphMode::Replay
            }
            Ok(Lookup::Capture) if capture => {
                let pool = self.pool.as_ref().ok_or_else(|| {
                    Error::Invariant("a video decoder captures into its graph pool".into())
                })?;
                job.capture(Capture {
                    functions: self.capture.clone(),
                    pool: pool.value.clone(),
                    peers: Vec::new(),
                    stream: self.stream.raw(),
                    device: self.device,
                    method: method(self)?,
                    breaks: None,
                });
                GraphMode::Capture
            }
            // A shape startup did not capture runs on the Python thread.
            _ => {
                job.call(method(self)?);
                GraphMode::Eager
            }
        };
        Ok(mode)
    }

    /// Decode and reconstruct media unit `unit` of a `frames`-frame video
    /// into the leading `written` frames of `out` (`[longest, height,
    /// width, 3]` uint8). `latent` is the request's complete packed latent,
    /// `shape` the unit's.
    #[expect(
        clippy::too_many_arguments,
        reason = "one unit's complete reconstruction inputs"
    )]
    pub(crate) fn decode(
        &mut self,
        slot: usize,
        latent: &StorageView,
        frames: u32,
        unit: u32,
        shape: UnitShape,
        longest: u32,
        written: u32,
        out: &StorageView,
        job: &Job,
    ) -> Result<GraphMode> {
        let canvas = shape.clip_output.1;
        let stream = self.stream.clone();
        let issue = job.on(Some(&stream));
        let latent_bytes = latent.numel() * latent.dtype().bytes() as u64;
        if !latent.is_contiguous()
            || latent.dtype() != DType::F32
            || latent_bytes > self.latent.numel() * 4
        {
            return Err(Error::Invalid(
                "a video latent exceeds the decoder's latent rows".into(),
            ));
        }
        let unit_rows = self.rows_of(frames, canvas, unit)?.to_vec();
        let rows = (unit_rows.len() / 8) as u64;
        let slot_overlap = self.overlap.slot("video_overlap", slot as u32)?;
        let overlap_bytes = self.retained.numel() * self.retained.dtype().bytes() as u64;
        // SAFETY: the decoder's clip rows, kept until it closes after its
        // stream drained; the forwards reading them run on `issue`'s stream,
        // the earlier ones before this write and this unit's after it.
        unsafe { self.rows.prefix(rows as i64)?.upload(&unit_rows, issue) }?;
        // The two copies write disjoint tensors, so they need no order
        // between them.
        let mut inputs = CopyBatch::new(SourceOrder::Stream);
        inputs.push(
            self.latent.address(),
            latent.address(),
            latent_bytes as usize,
        );
        inputs.push(
            self.retained.address(),
            slot_overlap.address(),
            overlap_bytes as usize,
        );
        // SAFETY: the forward's tensors and the slot's overlap row belong to
        // this runner and the copies stay inside them, in disjoint ranges;
        // the latent is leased until the batch commits and was waited for
        // on this stream.
        unsafe { issue.copy(&inputs) }?;

        let mode = self.forward(shape, rows, longest, false, job)?;

        let frame_bytes = u64::from(canvas.height) * u64::from(canvas.width) * 3;
        let mut outputs = CopyBatch::new(SourceOrder::Stream);
        outputs.push(
            slot_overlap.address(),
            self.retained.address(),
            overlap_bytes as usize,
        );
        outputs.push(
            out.address(),
            self.frames.address(),
            (u64::from(written) * frame_bytes) as usize,
        );
        // SAFETY: as above; `out` holds the unit's frames and the copies
        // follow the forward on the stream.
        unsafe { issue.copy(&outputs) }?;
        Ok(mode)
    }

    pub(crate) fn kernel_calls(&self) -> Vec<MethodCall> {
        self.class
            .kernel_calls(self.runners.iter().map(|(_, runner)| runner))
    }

    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        let graphs = self.graphs.close(aborted);
        graphs.and(
            self.class
                .close(aborted, self.runners.iter().map(|(_, runner)| runner)),
        )
    }
}

/// Decode and reconstruct this rank's media unit of a video round into its
/// row of the round's product.
pub(crate) fn decode_video(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    batch: &mut MediaBatch,
    index: usize,
    slot: usize,
    placement: &Placement<'_>,
    job: &Job,
) -> Result<()> {
    let call = &plan.calls[index];
    let range = decode_range(plan, call)?;
    let (frames, canvas, longest, units) = {
        let request = parts.video.request(slot, call.request_key)?;
        (
            request.frames,
            request.canvas,
            request.longest(),
            Arc::clone(&request.units),
        )
    };
    let total = units.len() as u32;
    if call.inputs.len() != 1 || call.outputs.len() != 1 {
        return Err(Error::Invalid(
            "media reconstruction requires one tensor input and output".into(),
        ));
    }
    if range.max_units == 0 || range.cursor + range.max_units > total {
        return Err(Error::Invalid(
            "media decoder assignment exceeds its media unit range".into(),
        ));
    }
    let rank = parts.rank;
    let decoder = parts
        .decoders
        .video
        .as_ref()
        .ok_or_else(|| Error::Invalid("this rank holds no video decoder".into()))?;
    let run = decoder.units(rank, range.cursor, range.max_units)?;
    if run.is_empty() {
        // This rank holds no unit of the round and joins no exchange.
        return Ok(());
    }
    if run.len() != 1 {
        return Err(Error::Unsupported(
            "video decoding reconstructs one media unit per rank and round".into(),
        ));
    }
    let unit = run.start;
    let clip_output: ClipOutput = parts.decoders.clip_output(frames, canvas, unit)?;
    let owners = parts.owners();
    let stream = stream_of(batch)?;
    let issue = job.on(Some(&stream));
    let latent = batch.read(owners.store, &call.inputs[0], call.call_id, issue)?;
    let row = u64::from(unit - range.cursor);
    let shape = vec![
        u64::from(range.max_units),
        u64::from(longest),
        u64::from(canvas.height),
        u64::from(canvas.width),
        3,
    ];
    let mut region: Vec<Range<u64>> = shape.iter().map(|&extent| 0..extent).collect();
    region[0] = row..row + 1;
    let write = bind_outputs(
        owners.store,
        &[OutputShape {
            product: call.outputs[0].clone(),
            shape,
            region: Some(region),
        }],
        batch.target(),
        placement,
    )?
    .into_iter()
    .next()
    .ok_or_else(|| Error::Invariant("a decoded unit row was not bound".into()))?;
    batch.batch.writes.push(write.clone());
    let view = write.view();
    let out = view.select(0, 0)?;
    let decoder = &mut parts
        .decoders
        .video
        .as_mut()
        .ok_or_else(|| Error::Invalid("this rank holds no video decoder".into()))?
        .runner;
    let written = units[unit as usize];
    let mode = decoder.decode_unit(
        slot,
        latent.value(),
        frames,
        unit,
        UnitShape {
            clip_output,
            first: unit == 0,
            last: unit + 1 == total,
            count: range.max_units,
        },
        longest,
        written,
        &out,
        job,
    )?;
    batch.record(mode);
    let frame_bytes = u64::from(canvas.height) * u64::from(canvas.width) * 3;
    let padding = u64::from(longest.saturating_sub(written)) * frame_bytes;
    if padding > 0 {
        // A short unit's row keeps zero frames after its own for its reader.
        // SAFETY: the padding lies inside the unit's bound product row,
        // written on the decoder's stream after the reconstruction.
        unsafe {
            issue.fill(Fill {
                address: out.address() + u64::from(written) * frame_bytes,
                value: 0,
                width: 1,
                count: padding as usize,
            })?;
        }
    }
    publish_bound(parts, batch, call, index, &write, true, issue)
}

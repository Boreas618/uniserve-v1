//! Audio decoding: a rank's decoder of one track length ([`AudioDecoder`]), its
//! forward captured once per track shape, and the decoding of the rank's audio
//! units into its span of the track ([`decode_audio`]).

use std::collections::BTreeMap;
use std::sync::Arc;

use cuda::{CopyBatch, DeviceId, SourceOrder, Stream};
use ipc::Batch as BatchPlan;

use super::DECODER;
use crate::cuda_graph::{GraphMode, GraphPool, GraphTable, Lookup};
use crate::executor::products::{OutputShape, Placement, bind_outputs};
use crate::executor::video::MediaBatch;
use crate::executor::video::{Parts, decode_range, publish_bound, stream_of};
use crate::ffi::DType;
use crate::lane::{Capture, Completion, Job};
use crate::memory::allocation::StorageView;
use crate::parallel::joins::GraphFunctions;
use crate::runner::methods::{Argument, MethodCall, Methods, RunnerClass};
use crate::runner::{RunnerSetup, device_buffer};
use crate::stream::Issue;
use crate::{Error, Result};

/// What fixes an audio decoding forward's computation: the track's samples
/// and which of how many units the forward decodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct TrackUnits {
    pub(crate) samples: u64,
    /// Latent frames of the track.
    pub(crate) latent_frames: u32,
    pub(crate) first: u32,
    pub(crate) count: u32,
    pub(crate) total: u32,
    /// Samples the units span.
    pub(crate) span: u64,
}

/// One rank's audio decoder.
///
/// A forward reads the request's audio latent and writes its samples at
/// fixed addresses, so one captured graph per [`TrackUnits`] serves every
/// request; the runner copies a request's values in and out natively.
pub(super) struct AudioDecoder {
    class: RunnerClass,
    pub(super) stream: Stream,
    pub(super) device: DeviceId,
    /// The runner of each track length, by its latent frames.
    runners: BTreeMap<u32, Methods>,
    /// `[rows, width]` FP32: a request's complete audio latent.
    latent: StorageView,
    /// `[samples, 2]` int16: the samples of the decoded units.
    samples: StorageView,
    pub(super) graphs: GraphTable<TrackUnits>,
    /// Captures a forward between its begin and end.
    capture: GraphFunctions,
    /// The graph pool captures allocate in; `None` serves forwards eagerly.
    pool: Option<Arc<GraphPool>>,
}

impl AudioDecoder {
    /// `latent` is the `[rows, width]` of the longest track's complete
    /// latent and `samples` that track's samples.
    pub(crate) fn new(
        setup: RunnerSetup<'_>,
        latent: [u64; 2],
        samples: u64,
        pool: Option<Arc<GraphPool>>,
    ) -> Result<Self> {
        let capture = GraphFunctions::import()?;
        Ok(Self {
            device: setup.device,
            runners: BTreeMap::new(),
            latent: device_buffer(
                setup.device,
                DType::F32,
                &[latent[0] as i64, latent[1] as i64],
            )?,
            samples: device_buffer(setup.device, DType::I16, &[samples as i64, 2])?,
            graphs: GraphTable::new(format!("{}.decoder", setup.path), capture.release.clone()),
            capture,
            pool,
            class: RunnerClass::new(DECODER, "decode", setup.arguments())?,
            stream: setup.stream.clone(),
        })
    }

    /// Construct the runner of tracks of `latent_frames` latent frames.
    ///
    /// Startup alone constructs (`DecoderRunners::prepare_startup`), for
    /// the track of every admitted frame count: a serving rank constructs
    /// no decoder runner, and a track without one is refused.
    pub(crate) fn prepare(&mut self, latent_frames: u32) -> Result<()> {
        if !self.runners.contains_key(&latent_frames) {
            let runner = self
                .class
                .construct(vec![Argument::Integers(vec![i64::from(latent_frames)])])?;
            self.runners.insert(latent_frames, runner);
        }
        Ok(())
    }

    /// The arguments of `DecoderRunner.decode` for `units` of a track whose
    /// complete latent has `rows` rows.
    fn arguments(&self, units: TrackUnits, rows: u64) -> Result<Vec<Argument>> {
        let runner = self.runners.get(&units.latent_frames).ok_or_else(|| {
            Error::Invalid("the audio track's length has no decoder runner".into())
        })?;
        Ok(vec![
            Argument::Handle(runner.runner().clone()),
            Argument::Tensor(self.latent.prefix(rows as i64)?),
            Argument::Tensor(self.samples.prefix(units.span as i64)?),
            Argument::Integer(units.samples as i64),
            Argument::Integer(i64::from(units.first)),
            Argument::Integer(i64::from(units.count)),
            Argument::Integer(i64::from(units.total)),
        ])
    }

    /// Record into `job` the decode of `units` of a zero track whose
    /// complete latent has `rows` rows, the forward as serving records it,
    /// so kernels compile before any capture, then the wait for the stream.
    pub(crate) fn warm(&mut self, units: TrackUnits, rows: u64, job: &Job) -> Result<()> {
        let stream = self.stream.clone();
        let issue = job.on(Some(&stream));
        // SAFETY: the decoder's own latent input, kept until it closes after
        // its stream drained; only its forwards read it, recorded behind the
        // fill on the same stream.
        unsafe { self.latent.zero(issue) }?;
        // No units are captured before every unit is warmed: the forward is
        // the eager call.
        self.forward(units, rows, false, job)?;
        issue.synchronize()
    }

    /// Record into `job` the capture of the forward of warmed `units`;
    /// whether it records one: a decoder without a graph pool serves every
    /// track eagerly, and captured units are not captured again.
    pub(crate) fn capture(&mut self, units: TrackUnits, rows: u64, job: &Job) -> Result<bool> {
        if self.pool.is_none() || self.graphs.contains(&units, None) {
            return Ok(false);
        }
        Ok(self.forward(units, rows, true, job)? == GraphMode::Capture)
    }

    /// Store the graph the job of [`Self::capture`] captured for `units`.
    pub(crate) fn complete_capture(
        &mut self,
        units: TrackUnits,
        completion: &mut Completion,
    ) -> Result<()> {
        self.graphs.insert(units, None, completion.captured()?)?;
        Ok(())
    }

    /// Record in `job` the forward of `units` of a track whose complete
    /// latent has `rows` rows, on the decoder's stream: the launch of the
    /// units' captured graph, or the eager call of `DecoderRunner.decode`,
    /// which serves units startup did not capture. With `capture`, units
    /// not captured yet are captured instead of called (startup's capture
    /// of warmed units): the job's completion carries the graph
    /// ([`Self::complete_capture`]). A capture is counted when its graph is
    /// stored.
    fn forward(
        &mut self,
        units: TrackUnits,
        rows: u64,
        capture: bool,
        job: &Job,
    ) -> Result<GraphMode> {
        let method = |decoder: &Self| -> Result<MethodCall> {
            Ok(MethodCall::new(
                &decoder.class.forward,
                decoder.arguments(units, rows)?,
            ))
        };
        let mode = match self.graphs.lookup(&units, None) {
            Ok(Lookup::Replay(graph)) => {
                job.on(Some(&self.stream)).launch(graph)?;
                GraphMode::Replay
            }
            Ok(Lookup::Capture) if capture => {
                let pool = self.pool.as_ref().ok_or_else(|| {
                    Error::Invariant("an audio decoder captures into its graph pool".into())
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
            // Units startup did not capture run on the builder thread.
            _ => {
                job.call(method(self)?);
                GraphMode::Eager
            }
        };
        Ok(mode)
    }

    /// Decode `units` of a track into `out` (`[span, 2]` int16). `latent`
    /// is the request's complete audio latent.
    pub(crate) fn decode(
        &mut self,
        latent: &StorageView,
        units: TrackUnits,
        out: &StorageView,
        job: &Job,
    ) -> Result<GraphMode> {
        let stream = self.stream.clone();
        let issue = job.on(Some(&stream));
        let rows = latent.shape().first().copied().unwrap_or(0).max(0) as u64;
        let latent_bytes = latent.numel() * latent.dtype().bytes() as u64;
        if !latent.is_contiguous()
            || latent.dtype() != DType::F32
            || latent_bytes > self.latent.numel() * 4
        {
            return Err(Error::Invalid(
                "an audio latent exceeds the decoder's latent rows".into(),
            ));
        }
        let mut input = CopyBatch::new(SourceOrder::Stream);
        input.push(
            self.latent.address(),
            latent.address(),
            latent_bytes as usize,
        );
        // SAFETY: the forward's tensors belong to this runner and the copy
        // stays inside them; the latent is leased until the batch commits
        // and was waited for on this stream.
        unsafe { issue.copy(&input) }?;

        let mode = self.forward(units, rows, false, job)?;

        let mut output = CopyBatch::new(SourceOrder::Stream);
        output.push(
            out.address(),
            self.samples.address(),
            (units.span * 4) as usize,
        );
        // SAFETY: as above; `out` holds the units' samples and the copy
        // follows the forward on the stream.
        unsafe { issue.copy(&output) }?;
        Ok(mode)
    }

    pub(crate) fn kernel_calls(&self) -> Vec<MethodCall> {
        self.class.kernel_calls(self.runners.values())
    }

    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        let graphs = self.graphs.close(aborted);
        graphs.and(self.class.close(aborted, self.runners.values()))
    }
}

/// Decode this rank's audio units into its span of the track.
pub(crate) fn decode_audio(
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
    let samples = parts.video.request(slot, call.request_key)?.audio_samples;
    if call.inputs.len() != 1 || call.outputs.len() != 1 {
        return Err(Error::Invalid(
            "media reconstruction requires one tensor input and output".into(),
        ));
    }
    let rank = parts.rank;
    let audio = parts
        .decoders
        .audio
        .as_ref()
        .ok_or_else(|| Error::Invalid("this rank holds no audio decoder".into()))?;
    let total = if audio.config.distribution.is_some() {
        (audio.config.ranks.len() * audio.config.units_per_rank.max(1)) as u32
    } else {
        1
    };
    let units = range.max_units.min(total).max(1);
    let run = audio.units(&range.ranks, rank, 0, units)?;
    if run.is_empty() {
        return Ok(());
    }
    let (latent_frames, spans) = parts.decoders.track(samples, units)?;
    let first = spans[run.start as usize].start;
    let last = spans[run.end as usize - 1].end;
    let owners = parts.owners();
    let stream = stream_of(batch)?;
    let issue = job.on(Some(&stream));
    let latent = batch.read(owners.store, &call.inputs[0], call.call_id, issue)?;
    let write = bind_outputs(
        owners.store,
        &[OutputShape {
            product: call.outputs[0].clone(),
            shape: vec![samples, 2],
            region: Some(vec![first..last, 0..2]),
        }],
        batch.target(),
        placement,
    )?
    .into_iter()
    .next()
    .ok_or_else(|| Error::Invariant("an audio span was not bound".into()))?;
    batch.batch.writes.push(write.clone());
    let audio = &mut parts
        .decoders
        .audio
        .as_mut()
        .ok_or_else(|| Error::Invalid("this rank holds no audio decoder".into()))?
        .runner;
    let mode = audio.decode_track(
        latent.value(),
        TrackUnits {
            samples,
            latent_frames,
            first: run.start,
            count: run.len() as u32,
            total: units,
            span: last - first,
        },
        &write.view(),
        job,
    )?;
    batch.record(mode);
    publish_bound(parts, batch, call, index, &write, true, issue)
}

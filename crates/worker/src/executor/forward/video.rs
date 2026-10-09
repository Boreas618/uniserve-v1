//! The video runtime: video requests, their denoising, decoding and
//! condition encoding on device ranks, and their media work on host ranks.
//!
//! [`VideoRuntime`] owns a rank's video components: the text encoder with
//! its vision tower, the denoiser runner with its latent pool and layouts,
//! the video and audio decoders, the condition latent encoders and the host
//! media (condition reading, codecs
//! and MP4 assembly). It admits each video request the rank serves when the
//! engine starts it, keeps the request's numerical size and layout while it
//! lives, and executes the request's media calls: device calls here, host
//! calls in [`super::media`]. Every batch runs through a
//! [`MediaBatch`](super::batch::MediaBatch): its inputs are imported before
//! it executes, its outputs are written and exported on the call's stream,
//! and its result is delivered once its device work and host tasks
//! completed.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;

use tvm_ffi::Any;
use uniserve_core::{Canvas, ComponentConfig, VideoCondition};
use uniserve_cuda::{Stream, memset_d8};
use uniserve_ipc::{
    Batch as BatchPlan, BufferAllocation, BufferId, Call, CallKind, MediaCall, NewRequest,
    RequestKey, TensorRef, WorkerInfo,
};

use super::batch::{MediaBatch, request_slots};
use crate::error::WorkerError;
use crate::executor::forward::placement::{
    OutputShape, Placement, ProductOwners, Reads, bind_completion, bind_outputs, export, skipped,
};
use crate::ffi::Region;
use crate::ffi::{DType, Device, Function, Handle, list};
use crate::host::mux::{MuxConfig, MuxFunctions, MuxSession};
use crate::host::reader::ConditionReader;
use crate::host::task::MediaTask;
use crate::host::{HostLane, HostTask};
use crate::model::video::VideoModel;
use crate::request::RequestPool;
use crate::request::video::{Admission, MediaPlan, VideoSize, condition_shapes, vision_spans};
use crate::runner::diffusion::video::VideoDenoiser;
use crate::runner::encoder::{EncoderConfig, PromptVision};
use crate::runner::video::{
    AudioDecoding, LatentEncoding, Segment, TextEncoding, TextSetup, VideoDecoding,
};
use crate::worker::RankStorage;
use crate::{Error, OutputPool, Result};

/// A placed video component on this rank.
pub(crate) struct Component<R> {
    pub(crate) name: String,
    pub(crate) config: ComponentConfig,
    pub(crate) device: Device,
    pub(crate) runner: R,
}

impl<R> Component<R> {
    /// This rank's media units of a round covering units `[cursor, cursor +
    /// count)`: all of them for an undistributed component, otherwise the
    /// rank's run of `units_per_rank` units from its position.
    pub(crate) fn units(&self, rank: usize, cursor: u32, count: u32) -> Result<Range<u32>> {
        units(&self.config, rank, cursor, count)
    }
}

/// See [`Component::units`].
pub(crate) fn units(
    config: &ComponentConfig,
    rank: usize,
    cursor: u32,
    count: u32,
) -> Result<Range<u32>> {
    if config.distribution.is_none() {
        return Ok(cursor..cursor + count);
    }
    let position = config
        .ranks
        .iter()
        .position(|&member| member == rank)
        .ok_or_else(|| Error::Invalid("media units assigned outside the component".into()))?
        as u32;
    let per_rank = config.units_per_rank as u32;
    let first = (position * per_rank).min(count);
    let last = (position * per_rank + per_rank).min(count);
    Ok(cursor + first..cursor + last)
}

/// One admitted video request on this rank.
pub(crate) struct VideoRequest {
    pub(crate) key: RequestKey,
    pub(crate) frames: u32,
    pub(crate) canvas: Canvas,
    /// The prompt's token ids.
    pub(crate) tokens: Arc<[i64]>,
    pub(crate) conditions: Arc<[VideoCondition]>,
    /// The request's layout on a denoising rank.
    pub(crate) admission: Option<Admission>,
    /// The denoiser's prepared layout id of `admission`.
    pub(crate) layout: Option<usize>,
    /// Frames of each media unit.
    pub(crate) units: Arc<[u32]>,
    /// Samples of the generated audio track.
    pub(crate) audio_samples: u64,
    /// The host task drawing the request's noise.
    pub(crate) preparation: Option<Arc<HostTask<MediaTask>>>,
    /// The request's MP4 on the muxer rank.
    pub(crate) mux: Option<Arc<MuxSession>>,
}

impl VideoRequest {
    /// The longest media unit's frames.
    pub(crate) fn longest(&self) -> u32 {
        self.units.iter().copied().max().unwrap_or(0)
    }
}

/// The host media components of a rank.
pub(crate) struct HostMedia {
    pub(crate) reader: Option<Arc<ConditionReader>>,
    pub(crate) encode_video: Function,
    pub(crate) encode_audio: Function,
    pub(crate) mux: MuxFunctions,
    pub(crate) lane: HostLane<MediaTask>,
    /// Host component placements, by name.
    pub(crate) components: BTreeMap<String, ComponentConfig>,
}

/// A rank's video runtime.
pub(crate) struct VideoRuntime {
    pub(crate) model: VideoModel,
    pub(crate) plan: MediaPlan,
    pub(crate) rank: usize,
    /// Admitted requests by slot.
    pub(crate) requests: HashMap<usize, VideoRequest>,
    pub(crate) text: Option<Component<TextEncoding>>,
    /// The text encoder's graph pool, when graphs are captured.
    pub(crate) text_pool: Option<Arc<crate::runner::GraphPool<Handle>>>,
    pub(crate) denoiser: Option<Component<VideoDenoiser>>,
    pub(crate) decoder: Option<Component<VideoDecoding>>,
    pub(crate) audio: Option<Component<AudioDecoding>>,
    pub(crate) latent: Option<Component<LatentEncoding>>,
    pub(crate) host: Option<HostMedia>,
    /// Draws requests' noise beside device work.
    pub(crate) noise: Option<HostLane<MediaTask>>,
    pub(crate) outputs: OutputPool,
    /// Product names of every component's outputs, by output index.
    pub(crate) products: HashMap<String, Vec<String>>,
    /// The first rank of each component reports its host results.
    pub(crate) output_ranks: HashMap<String, usize>,
}

/// The parts of the backend a media batch executes with.
pub(crate) struct Parts<'a> {
    pub(crate) requests: &'a mut RequestPool,
    pub(crate) storage: &'a RankStorage,
    pub(crate) video: &'a mut VideoRuntime,
    pub(crate) info: &'a WorkerInfo,
}

impl<'a> Parts<'a> {
    /// The product owners, borrowed for the parts' lifetime rather than
    /// from the parts themselves.
    pub(crate) fn owners(&self) -> ProductOwners<'a> {
        let storage: &'a RankStorage = self.storage;
        ProductOwners {
            store: &storage.tensors,
            transports: &storage.transports,
            events: &storage.events,
        }
    }
}

fn media_kind(call: &Call) -> MediaCall {
    match call.code {
        CallKind::Media(kind) => kind,
        _ => MediaCall::Denoising,
    }
}

impl VideoRuntime {
    /// The name of a call's output product.
    pub(crate) fn product_name(&self, call: &Call, output: &TensorRef) -> Result<&str> {
        self.products
            .get(&call.component)
            .and_then(|names| names.get(output.output_index as usize))
            .map(String::as_str)
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "component {} declares no output {}",
                    call.component, output.output_index
                ))
            })
    }

    /// Whether this rank reports a component's host results.
    pub(crate) fn reports(&self, component: &str) -> bool {
        self.output_ranks
            .get(component)
            .is_none_or(|&rank| rank == self.rank)
    }

    /// Admit a started video request: resolve its size and, on a denoising
    /// rank, its layout, and start drawing its noise.
    pub(crate) fn admit(&mut self, admission: &NewRequest) -> Result<()> {
        let (Some(media), Some(video)) = (&admission.diffusion, &admission.video) else {
            return Ok(());
        };
        let slot = admission.request_pool_idx as usize;
        let canvas = Canvas {
            width: media.width,
            height: media.height,
        };
        if (video.task == uniserve_core::VideoTask::T2va) != video.conditions.is_empty() {
            return Err(Error::Invalid(
                "a video request's task disagrees with its conditions".into(),
            ));
        }
        if !self.plan.frame_counts.contains(&media.num_frames)
            || !self.plan.canvases.contains(&canvas)
        {
            return Err(Error::Invalid(
                "media input exceeds the worker frame, canvas, conditioning or condition capacity"
                    .into(),
            ));
        }
        let units: Arc<[u32]> = self.model.unit_frames(media.num_frames)?.into();
        if media.video_units as usize != units.len()
            || media.num_inference_steps != self.model.facts.num_steps
        {
            return Err(Error::Invalid(
                "video request has invalid computation bounds".into(),
            ));
        }
        let conditions: Arc<[VideoCondition]> = video.conditions.clone().into();
        let size = VideoSize {
            frames: media.num_frames,
            canvas,
            tokens: admission.prompt_token_ids.len() as u32,
            conditions: condition_shapes(&conditions),
            spans: vision_spans(&video.text_tags).into(),
        };
        let mut request = VideoRequest {
            key: admission.request_key,
            frames: media.num_frames,
            canvas,
            tokens: admission
                .prompt_token_ids
                .iter()
                .map(|&token| i64::from(token))
                .collect(),
            conditions,
            admission: None,
            layout: None,
            audio_samples: self.model.track_samples(media.num_frames)?,
            units,
            preparation: None,
            mux: None,
        };
        if let Some(denoiser) = &mut self.denoiser {
            let admitted = self.plan.admit(&self.model, size)?;
            let id = denoiser.runner.acquire(&self.model, admitted.layout)?;
            request.layout = Some(id);
            let lane = self
                .noise
                .as_ref()
                .ok_or_else(|| Error::Invariant("a denoising rank has no noise lane".into()))?;
            let task = lane.reserve()?;
            let arguments = [
                Any::from(slot as i64),
                Any::from(id as i64),
                admitted.size.descriptor(),
                admitted.size.conditions_descriptor(),
                admitted.size.spans_descriptor(),
                // The seed's bits; the binding reads them as unsigned.
                Any::from(media.seed as i64),
            ]
            .into_iter()
            .map(Handle::new)
            .collect();
            let configured = task
                .configure(MediaTask::Prepare {
                    function: denoiser.runner.preparation(),
                    arguments,
                })
                .and_then(|()| task.submit());
            if let Err(error) = configured {
                let _ = task.cancel(true);
                denoiser.runner.release(id)?;
                return Err(error);
            }
            request.preparation = Some(task);
            request.admission = Some(admitted);
        }
        if let Some(previous) = self.requests.insert(slot, request) {
            self.retire(previous)?;
        }
        Ok(())
    }

    fn retire(&mut self, request: VideoRequest) -> Result<()> {
        if let Some(task) = &request.preparation
            && !task.completion.done()
        {
            // A drawing task still writes the slot's sources: wait for it.
            let _ = task.cancel(false);
            let _ = task.completion.wait(None);
        }
        if let (Some(denoiser), Some(id)) = (&mut self.denoiser, request.layout) {
            denoiser.runner.release(id)?;
        }
        Ok(())
    }

    pub(crate) fn request(&self, slot: usize, key: RequestKey) -> Result<&VideoRequest> {
        self.requests
            .get(&slot)
            .filter(|request| request.key == key)
            .ok_or_else(|| Error::Invalid("video call has no admitted media dimensions".into()))
    }

    /// The MP4 session of a request on the muxer rank, created on first use.
    pub(crate) fn mux_session(&mut self, slot: usize, key: RequestKey) -> Result<Arc<MuxSession>> {
        let functions = self
            .host
            .as_ref()
            .ok_or_else(|| Error::Invalid("this rank holds no muxer".into()))?
            .mux
            .clone();
        let frame_rate = self.model.facts.frame_rate;
        let audio_rate = self.model.facts.sample_rate;
        let request = self
            .requests
            .get_mut(&slot)
            .filter(|request| request.key == key)
            .ok_or_else(|| Error::Invalid("video call has no admitted media dimensions".into()))?;
        if let Some(session) = &request.mux {
            return Ok(Arc::clone(session));
        }
        let config = MuxConfig::new(
            request.canvas.width,
            request.canvas.height,
            frame_rate,
            audio_rate,
            Arc::clone(&request.units),
        )?;
        let session = Arc::new(MuxSession::new(config, functions));
        request.mux = Some(Arc::clone(&session));
        Ok(session)
    }

    /// The stream and device a batch of `kind` on `component` runs on.
    pub(crate) fn placement(
        &self,
        kind: MediaCall,
        component: &str,
    ) -> Result<(Device, Option<Stream>)> {
        let borrowed = |stream: &Stream| -> Result<Stream> {
            // SAFETY: the runner owns no stream: the worker's lanes own it and
            // close after the runtime (see `Worker::close`).
            Ok(unsafe { Stream::borrowed(stream.device(), stream.raw()) }?)
        };
        match kind {
            MediaCall::TextEncoding | MediaCall::VisionEncoding => {
                let text = self
                    .text
                    .as_ref()
                    .ok_or_else(|| Error::Invalid("this rank holds no text encoder".into()))?;
                Ok((text.device, Some(borrowed(text.runner.stream())?)))
            }
            MediaCall::LatentPreparation | MediaCall::Denoising => {
                let denoiser = self
                    .denoiser
                    .as_ref()
                    .filter(|denoiser| denoiser.name == component);
                let denoiser = denoiser
                    .ok_or_else(|| Error::Invalid("this rank holds no video denoiser".into()))?;
                Ok((denoiser.device, Some(borrowed(denoiser.runner.stream())?)))
            }
            MediaCall::VideoDecoding => {
                let decoder = self
                    .decoder
                    .as_ref()
                    .ok_or_else(|| Error::Invalid("this rank holds no video decoder".into()))?;
                Ok((decoder.device, Some(borrowed(decoder.runner.stream())?)))
            }
            MediaCall::AudioDecoding => {
                let audio = self
                    .audio
                    .as_ref()
                    .ok_or_else(|| Error::Invalid("this rank holds no audio decoder".into()))?;
                Ok((audio.device, Some(borrowed(audio.runner.stream())?)))
            }
            MediaCall::LatentEncoding => {
                let latent = self
                    .latent
                    .as_ref()
                    .ok_or_else(|| Error::Invalid("this rank holds no latent encoder".into()))?;
                Ok((latent.device, Some(borrowed(latent.runner.stream())?)))
            }
            _ => Ok((Device::Cpu, None)),
        }
    }
}

/// Open a video batch: its output lease on the stream of its kind, its
/// calls' pending results, and the calls its false predicates skip.
/// `borrowed` names the inputs host codecs read in place.
pub(crate) fn open<'b>(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    borrowed: HashSet<BufferId>,
    state: &'b mut Option<MediaBatch>,
) -> Result<&'b mut MediaBatch> {
    let first = plan
        .calls
        .first()
        .ok_or_else(|| Error::Invariant("a video batch has calls".into()))?;
    let (device, stream) = parts.video.placement(media_kind(first), &first.component)?;
    let batch = state.insert(MediaBatch::new(
        device,
        stream,
        &mut parts.video.outputs,
        plan.calls.len(),
        Reads::new(borrowed),
    )?);
    let video = &*parts.video;
    batch.open_calls(plan, parts.requests, |component| video.reports(component))?;
    let skipped = skipped(
        &parts.storage.tensors,
        plan,
        batch.device,
        batch.stream.as_ref(),
    )?;
    batch.skip(&skipped);
    Ok(batch)
}

/// Execute a batch of device video calls and commit it.
pub(crate) fn execute(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    borrowed: HashSet<BufferId>,
    state: &mut Option<MediaBatch>,
) -> std::result::Result<(), WorkerError> {
    let batch = open(parts, plan, borrowed, state)?;

    let slots = request_slots(plan, parts.requests)?;
    let allocations: HashMap<BufferId, BufferAllocation> = plan
        .buffer_allocations
        .iter()
        .map(|allocation| (allocation.buffer, *allocation))
        .collect();
    let placement = Placement {
        allocations: &allocations,
        slots: &slots,
    };
    for index in 0..plan.calls.len() {
        let call = &plan.calls[index];
        let slot = slots[&call.request_key];
        let owners = parts.owners();
        let completion = bind_completion(owners.store, call, batch.device, &slots)?;
        batch.calls[index].completion = completion;
        if batch.calls[index].skipped {
            continue;
        }
        match call.code {
            CallKind::Media(MediaCall::TextEncoding) => {
                encode_text(parts, plan, batch, index, slot, &placement)?
            }
            CallKind::Media(MediaCall::VisionEncoding) => {
                encode_vision(parts, plan, batch, index, slot, &placement)?
            }
            CallKind::Media(MediaCall::LatentPreparation) => {
                prepare(parts, plan, batch, index, slot)?
            }
            CallKind::Media(MediaCall::Denoising) => {
                denoise(parts, plan, batch, index, slot, &placement)?
            }
            CallKind::Media(MediaCall::VideoDecoding) => {
                decode_video(parts, plan, batch, index, slot, &placement)?
            }
            CallKind::Media(MediaCall::AudioDecoding) => {
                decode_audio(parts, plan, batch, index, slot, &placement)?
            }
            CallKind::Media(MediaCall::LatentEncoding) => {
                encode_conditions(parts, plan, batch, index, slot, &placement)?
            }
            kind => {
                return Err(Error::Unsupported(format!(
                    "{} is not a device video call",
                    kind.as_str()
                ))
                .into());
            }
        }
    }
    let owners = parts.owners();
    let trajectories = parts
        .video
        .denoiser
        .as_mut()
        .map(|denoiser| &mut denoiser.runner as &mut dyn super::batch::Trajectories);
    batch.commit(plan, owners, parts.requests, trajectories)?;
    Ok(())
}

fn latent_params<'a>(plan: &'a BatchPlan, call: &Call) -> Result<&'a uniserve_ipc::LatentParams> {
    plan.latent_params
        .iter()
        .find(|params| params.request_key == call.request_key && params.call_id == call.call_id)
        .ok_or_else(|| Error::Invalid("a video trajectory call has no latent pages".into()))
}

fn decode_range<'a>(plan: &'a BatchPlan, call: &Call) -> Result<&'a uniserve_ipc::DecodeRange> {
    plan.decode_ranges
        .iter()
        .find(|range| range.request_key == call.request_key && range.call_id == call.call_id)
        .ok_or_else(|| Error::Invalid("a media round has no unit range".into()))
}

/// A borrowed view of the batch's stream.
fn stream_of(batch: &MediaBatch) -> Result<Stream> {
    let stream = batch
        .stream
        .as_ref()
        .ok_or_else(|| Error::Invariant("a device media call has no stream".into()))?;
    // SAFETY: the lanes own the stream for the runtime's lifetime.
    Ok(unsafe { Stream::borrowed(stream.device(), stream.raw()) }?)
}

/// The most elements a product holds along one dimension.
fn extent(dim: &uniserve_ipc::DimBound) -> u64 {
    match dim {
        uniserve_ipc::DimBound::Static(extent) | uniserve_ipc::DimBound::Device { max: extent } => {
            u64::from(*extent)
        }
    }
}

/// Rows' width of a two-dimensional product.
fn row_width(output: &TensorRef) -> Result<u64> {
    output
        .shape_bound
        .dims
        .get(1)
        .map(extent)
        .ok_or_else(|| Error::Invalid("the product is not rows".into()))
}

/// The patch grids of a request's vision blocks, images and videos apart,
/// each in request order as the text encoder places them.
fn vision_grids(conditions: &[VideoCondition]) -> (Vec<[u32; 3]>, Vec<[u32; 3]>) {
    let mut images = Vec::new();
    let mut videos = Vec::new();
    for condition in conditions {
        let Some(vision) = &condition.vision else {
            continue;
        };
        let grid = [vision.grid.t, vision.grid.h, vision.grid.w];
        match condition.media {
            uniserve_core::ConditionMedia::Video { .. } => videos.push(grid),
            _ => images.push(grid),
        }
    }
    (images, videos)
}

/// Bind a call's single output as rows of `shape` on every rank that
/// computes it.
fn bind_rows(
    parts: &Parts<'_>,
    batch: &mut MediaBatch,
    index: usize,
    output: &TensorRef,
    shape: Vec<u64>,
    placement: &Placement<'_>,
) -> Result<crate::ProductBuffer> {
    let owners = parts.owners();
    let write = bind_outputs(
        owners.store,
        &[OutputShape {
            product: output.clone(),
            shape,
            region: None,
        }],
        batch.device,
        placement,
    )?
    .into_iter()
    .next()
    .ok_or_else(|| Error::Invariant("an encoder product was not bound".into()))?;
    batch.calls[index].writes.push(write.clone());
    Ok(write)
}

/// Publish a call's output, written at `shape` from its leading elements,
/// and export it to its readers.
fn publish(
    parts: &Parts<'_>,
    batch: &mut MediaBatch,
    call: &Call,
    index: usize,
    write: &crate::ProductBuffer,
    shape: &[u64],
) -> Result<()> {
    let owners = parts.owners();
    let stream = stream_of(batch)?;
    owners.store.produced(write, shape, Some(&stream), None)?;
    let (exported, locations) = export(
        owners,
        call,
        &call.outputs[0],
        write,
        false,
        None,
        Some(&stream),
    )?;
    batch.calls[index].publish(write.id(), locations, exported)?;
    Ok(())
}

/// Publish a call's first output, whose bound view the producer filled, and
/// export it to its readers (`host` readers read it from host memory).
///
/// The bound view is this rank's share of the product: its region when
/// other ranks hold the rest, the whole product when the round assigns
/// this rank all of it. Publishing the view itself is right in both cases;
/// the store tracks a share that covers the product as a whole product.
fn publish_bound(
    parts: &Parts<'_>,
    batch: &mut MediaBatch,
    call: &Call,
    index: usize,
    write: &crate::ProductBuffer,
    host: bool,
) -> Result<()> {
    let owners = parts.owners();
    let stream = stream_of(batch)?;
    owners
        .store
        .write(write, &write.view(), Some(&stream), None)?;
    let (exported, locations) = export(
        owners,
        call,
        &call.outputs[0],
        write,
        host,
        None,
        Some(&stream),
    )?;
    batch.calls[index].publish(write.id(), locations, exported)?;
    Ok(())
}

/// Encode a request's prompt into its `conditioning` rows, splicing the
/// vision features of its condition media in at their placeholders when the
/// call reads them. Every rank of the text encoder computes the features
/// and publishes its copy.
fn encode_text(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    batch: &mut MediaBatch,
    index: usize,
    slot: usize,
    placement: &Placement<'_>,
) -> Result<()> {
    let call = &plan.calls[index];
    if call.outputs.len() != 1 || call.inputs.len() > 1 {
        return Err(Error::Invalid(
            "text encoding reads at most one vision product and writes its conditioning".into(),
        ));
    }
    let (tokens, conditions) = {
        let request = parts.video.request(slot, call.request_key)?;
        (Arc::clone(&request.tokens), Arc::clone(&request.conditions))
    };
    if tokens.is_empty() {
        return Err(Error::Invalid(
            "text conditioning requires admitted prompt tokens".into(),
        ));
    }
    let width = row_width(&call.outputs[0])?;
    let stream = stream_of(batch)?;
    let visual = call
        .inputs
        .first()
        .map(|input| {
            batch.inputs.read(
                parts.owners().store,
                input,
                call.call_id,
                batch.device,
                Some(&stream),
            )
        })
        .transpose()?;
    let write = bind_rows(
        parts,
        batch,
        index,
        &call.outputs[0],
        vec![tokens.len() as u64, width],
        placement,
    )?;

    let text = &mut parts
        .video
        .text
        .as_mut()
        .ok_or_else(|| Error::Invalid("this rank holds no text encoder".into()))?
        .runner;
    let arena = text.text.products().clone();
    let out = Region::within(&arena, &write.view())?;
    let (image_grids, video_grids) = vision_grids(&conditions);
    let rows = visual
        .as_ref()
        .map(|read| Region::within(&arena, read.value()))
        .transpose()?;
    let vision = rows.as_ref().map(|rows| PromptVision {
        image_grids: &image_grids,
        video_grids: &video_grids,
        rows,
    });
    let (shape, _) = text.text.encode_text(&tokens, &out, vision)?;
    if shape.as_deref() != Some(&[tokens.len() as i64, width as i64][..]) {
        return Err(Error::Invalid(
            "prompt features disagree with the request's prompt and the conditioning width".into(),
        ));
    }
    publish(
        parts,
        batch,
        call,
        index,
        &write,
        &[tokens.len() as u64, width],
    )
}

/// Encode the vision blocks of a request's condition media into one
/// `vision_features` row per vision placeholder, in request order.
fn encode_vision(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    batch: &mut MediaBatch,
    index: usize,
    slot: usize,
    placement: &Placement<'_>,
) -> Result<()> {
    let call = &plan.calls[index];
    if call.inputs.len() != 1 || call.outputs.len() != 1 {
        return Err(Error::Invalid(
            "vision encoding requires one input and one output".into(),
        ));
    }
    let conditions = Arc::clone(&parts.video.request(slot, call.request_key)?.conditions);
    let blocks: Vec<([u32; 3], u64)> = conditions
        .iter()
        .filter_map(|condition| condition.vision.as_ref())
        .map(|vision| {
            (
                [vision.grid.t, vision.grid.h, vision.grid.w],
                u64::from(vision.tokens),
            )
        })
        .collect();
    if blocks.is_empty() {
        return Err(Error::Invalid(
            "vision encoding requires a vision block".into(),
        ));
    }
    let width = row_width(&call.outputs[0])?;
    let tokens: u64 = blocks.iter().map(|block| block.1).sum();
    let stream = stream_of(batch)?;
    let patches = batch.inputs.read(
        parts.owners().store,
        &call.inputs[0],
        call.call_id,
        batch.device,
        Some(&stream),
    )?;
    let write = bind_rows(
        parts,
        batch,
        index,
        &call.outputs[0],
        vec![tokens, width],
        placement,
    )?;

    let text = &mut parts
        .video
        .text
        .as_mut()
        .ok_or_else(|| Error::Invalid("this rank holds no text encoder".into()))?
        .runner;
    let vision = text
        .vision
        .as_mut()
        .ok_or_else(|| Error::Invalid("this rank's text encoder reads no vision".into()))?;
    let arena = vision.products().clone();
    let rows = Region::within(&arena, patches.value())?;
    let product = Region::within(&arena, &write.view())?;
    // Each block's features are consecutive rows of the product.
    let row_bytes = width * product.dtype().bytes() as u64;
    let mut start = 0;
    let mut out = Vec::with_capacity(blocks.len());
    for &(_, count) in &blocks {
        out.push(Region::new(
            product.offset() + start * row_bytes,
            product.dtype(),
            vec![count as i64, width as i64],
        )?);
        start += count;
    }
    let grids: Vec<[u32; 3]> = blocks.iter().map(|block| block.0).collect();
    let encoded = vision.encode_patches(&rows, &grids, &out)?;
    let planned: Vec<Vec<i64>> = blocks
        .iter()
        .map(|block| vec![block.1 as i64, width as i64])
        .collect();
    if encoded.shapes != planned {
        return Err(Error::Invalid(
            "vision features disagree with the request's placeholders".into(),
        ));
    }
    publish(parts, batch, call, index, &write, &[tokens, width])
}

/// Initialize a request's trajectory: its prepared sources, its refined
/// prompt and its encoded conditions.
fn prepare(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    batch: &mut MediaBatch,
    index: usize,
    slot: usize,
) -> Result<()> {
    let call = &plan.calls[index];
    let params = latent_params(plan, call)?;
    let (id, preparation, conditioned, size) = {
        let request = parts.video.request(slot, call.request_key)?;
        let id = request
            .layout
            .ok_or_else(|| Error::Invariant("a denoising request has no layout".into()))?;
        (
            id,
            request.preparation.clone(),
            !request.conditions.is_empty(),
            request
                .admission
                .as_ref()
                .map(|admission| admission.size.clone()),
        )
    };
    if call.inputs.is_empty() || (call.inputs.len() > 1) != conditioned {
        return Err(Error::Invalid(
            "video preparation reads its conditioning and its conditions".into(),
        ));
    }
    if let Some(task) = preparation {
        match task.completion.wait(None) {
            Some(crate::Outcome::Success(_)) => {}
            Some(crate::Outcome::Failed(error)) => {
                return Err(Error::Invalid(format!(
                    "video request preparation failed: {error}"
                )));
            }
            _ => {
                return Err(Error::Invalid(
                    "video request preparation was cancelled".into(),
                ));
            }
        }
    }
    let store = parts.owners().store;
    let stream = stream_of(batch)?;
    let device = batch.device;
    let features =
        batch
            .inputs
            .read(store, &call.inputs[0], call.call_id, device, Some(&stream))?;
    let conditions: Vec<Arc<crate::ProductRead>> = call.inputs[1..]
        .iter()
        .map(|input| {
            batch
                .inputs
                .read(store, input, call.call_id, device, Some(&stream))
        })
        .collect::<Result<_>>()?;
    let denoiser = parts
        .video
        .denoiser
        .as_mut()
        .ok_or_else(|| Error::Invalid("this rank holds no video denoiser".into()))?;
    let update = denoiser.runner.initialize(id, slot, call, params)?;
    let value = features.value();
    let count = size.as_ref().map_or(0, |size| u64::from(size.tokens));
    if value.shape().first().copied() != Some(count as i64) {
        return Err(Error::Invalid(
            "prompt features disagree with the request's prompt".into(),
        ));
    }
    denoiser.runner.condition(id, slot, value, count)?;
    if conditioned {
        let size =
            size.ok_or_else(|| Error::Invariant("a conditioned request has no size".into()))?;
        let request = parts.video.request(slot, call.request_key)?;
        let latents = condition_latents(&request.conditions, &conditions)?;
        let arguments = [
            size.descriptor(),
            size.conditions_descriptor(),
            size.spans_descriptor(),
        ];
        let denoiser = parts
            .video
            .denoiser
            .as_ref()
            .ok_or_else(|| Error::Invalid("this rank holds no video denoiser".into()))?;
        denoiser
            .runner
            .encode_conditions(id, slot, &arguments, latents)?;
    }
    batch.calls[index].latent = Some(update);
    Ok(())
}

/// Rows of one condition read: `(read, first row, rows)`.
type RowSpan = (usize, u64, u64);

/// Locate every condition latent in latent preparation's condition reads.
///
/// `conditions` are each condition's `(visual rows, audio rows)` in request
/// order. The reads are the visual rounds' rows in round order (`rounds`
/// holds each round's row count), then, when any condition carries audio,
/// one read of every track's rows (`tracks`). The visual rounds hold the
/// conditions' visual rows back to back, and a round ends wherever its
/// units do, so a condition's rows may continue into the next round.
///
/// Returns the latents in request order, a condition's visual rows before
/// its audio rows, each as the spans that hold its rows in row order.
fn condition_spans(
    conditions: &[(u64, u64)],
    rounds: &[u64],
    tracks: Option<u64>,
) -> Result<Vec<Vec<RowSpan>>> {
    let mismatch =
        || Error::Invalid("condition latents disagree with the request's planned rows".into());
    let visual: u64 = conditions.iter().map(|rows| rows.0).sum();
    let audio: u64 = conditions.iter().map(|rows| rows.1).sum();
    if rounds.iter().sum::<u64>() != visual || tracks.unwrap_or(0) != audio {
        return Err(mismatch());
    }
    // The audio read follows the visual rounds.
    let track_read = rounds.len();
    let mut result = Vec::new();
    let (mut round, mut row, mut audio_row) = (0usize, 0u64, 0u64);
    for &(visual_rows, audio_rows) in conditions {
        if visual_rows > 0 {
            let mut spans = Vec::new();
            let mut needed = visual_rows;
            while needed > 0 {
                let available = *rounds.get(round).ok_or_else(mismatch)? - row;
                let taken = available.min(needed);
                if taken > 0 {
                    spans.push((round, row, taken));
                }
                needed -= taken;
                row += taken;
                if taken == available {
                    round += 1;
                    row = 0;
                }
            }
            result.push(spans);
        }
        if audio_rows > 0 {
            result.push(vec![(track_read, audio_row, audio_rows)]);
            audio_row += audio_rows;
        }
    }
    Ok(result)
}

/// The condition latents of latent preparation's condition `reads`, each a
/// list of row tensors that together hold its rows (see
/// [`condition_spans`]).
fn condition_latents(
    conditions: &[VideoCondition],
    reads: &[Arc<crate::ProductRead>],
) -> Result<Vec<Any>> {
    let audio = conditions.iter().any(|condition| condition.audio_rows > 0);
    let (rounds, tracks) = match (audio, reads.split_last()) {
        (true, Some((tracks, rounds))) => (rounds, Some(tracks)),
        (true, None) => {
            return Err(Error::Invalid(
                "condition latents lack their audio rows".into(),
            ));
        }
        (false, _) => (reads, None),
    };
    let rows = |read: &Arc<crate::ProductRead>| {
        read.value().shape().first().copied().unwrap_or(0).max(0) as u64
    };
    let planned: Vec<(u64, u64)> = conditions
        .iter()
        .map(|condition| (condition.video_rows(), u64::from(condition.audio_rows)))
        .collect();
    let round_rows: Vec<u64> = rounds.iter().map(rows).collect();
    condition_spans(&planned, &round_rows, tracks.map(rows))?
        .into_iter()
        .map(|spans| {
            let parts = spans
                .into_iter()
                .map(|(read, first, count)| {
                    let rows = reads[read].value().narrow(0, first as i64, count as i64)?;
                    Ok(Any::from(rows.to_tensor()?))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Any::from(list(parts)))
        })
        .collect()
}

/// Advance a request's trajectory by one step; the final step publishes the
/// rank's rows of each modality's latents.
fn denoise(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    batch: &mut MediaBatch,
    index: usize,
    slot: usize,
    placement: &Placement<'_>,
) -> Result<()> {
    let call = &plan.calls[index];
    let params = latent_params(plan, call)?;
    let id = parts
        .video
        .request(slot, call.request_key)?
        .layout
        .ok_or_else(|| Error::Invariant("a denoising request has no layout".into()))?;
    let num_steps = parts.video.model.facts.num_steps;
    let names: Vec<String> = call
        .outputs
        .iter()
        .map(|output| parts.video.product_name(call, output).map(str::to_owned))
        .collect::<Result<_>>()?;
    let denoiser = parts
        .video
        .denoiser
        .as_mut()
        .ok_or_else(|| Error::Invalid("this rank holds no video denoiser".into()))?;
    let (update, _) = denoiser.runner.denoise(id, slot, call, params)?;
    batch.calls[index].latent = Some(update);
    if call.outputs.is_empty() {
        return Ok(());
    }
    if params.start_step + params.step_count != num_steps {
        return Err(Error::Invalid(
            "final latent products require completed denoising".into(),
        ));
    }
    let samples = denoiser.runner.final_samples(id)?;
    let rows = denoiser.runner.sample_rows(id)?;
    let modalities = parts.video.model.facts.modalities.clone();
    let owners = parts.owners();
    let stream = stream_of(batch)?;
    for (output, name) in call.outputs.iter().zip(&names) {
        let modality = modalities
            .iter()
            .position(|modality| format!("{modality}_latents") == *name)
            .ok_or_else(|| Error::Invalid(format!("the denoiser publishes no {name}")))?;
        let (Some(sample), sample_rows) = (&samples[modality], rows[modality]) else {
            // This rank holds none of the modality's rows.
            continue;
        };
        let shape = vec![sample_rows.rows, sample_rows.width];
        let region = Some(vec![
            sample_rows.local.0..sample_rows.local.1,
            0..sample_rows.width,
        ]);
        let bound = bind_outputs(
            owners.store,
            &[OutputShape {
                product: output.clone(),
                shape,
                region,
            }],
            batch.device,
            placement,
        )?;
        let write = bound
            .into_iter()
            .next()
            .ok_or_else(|| Error::Invariant("a latent product was not bound".into()))?;
        batch.calls[index].writes.push(write.clone());
        owners.store.write(&write, sample, Some(&stream), None)?;
        let (exported, locations) =
            export(owners, call, output, &write, false, None, Some(&stream))?;
        batch.calls[index].publish(write.id(), locations, exported)?;
    }
    Ok(())
}

/// Decode and reconstruct this rank's media unit of a video round into its
/// row of the round's product.
fn decode_video(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    batch: &mut MediaBatch,
    index: usize,
    slot: usize,
    placement: &Placement<'_>,
) -> Result<()> {
    let call = &plan.calls[index];
    let range = decode_range(plan, call)?;
    let (frames, canvas, longest, total) = {
        let request = parts.video.request(slot, call.request_key)?;
        (
            request.frames,
            request.canvas,
            request.longest(),
            request.units.len() as u32,
        )
    };
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
    let rank = parts.video.rank;
    let decoder = parts
        .video
        .decoder
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
    let segment: Segment = parts.video.model.segment(frames, canvas, unit)?;
    let owners = parts.owners();
    let stream = stream_of(batch)?;
    let latent = batch.inputs.read(
        owners.store,
        &call.inputs[0],
        call.call_id,
        batch.device,
        Some(&stream),
    )?;
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
        batch.device,
        placement,
    )?
    .into_iter()
    .next()
    .ok_or_else(|| Error::Invariant("a decoded unit row was not bound".into()))?;
    batch.calls[index].writes.push(write.clone());
    let view = write.view();
    let out = view.select(0, 0)?;
    let decoder = &mut parts
        .video
        .decoder
        .as_mut()
        .ok_or_else(|| Error::Invalid("this rank holds no video decoder".into()))?
        .runner;
    let (written, _) = decoder.decode(
        slot,
        latent.value(),
        frames,
        canvas,
        segment,
        unit,
        range.max_units,
        &out,
    )?;
    let frame_bytes = u64::from(canvas.height) * u64::from(canvas.width) * 3;
    let padding = u64::from(longest.saturating_sub(written)) * frame_bytes;
    if padding > 0 {
        // A short unit's row keeps zero frames after its own for its reader.
        // SAFETY: the padding lies inside the unit's bound product row,
        // written on the decoder's stream after the reconstruction.
        unsafe {
            memset_d8(
                out.address() + u64::from(written) * frame_bytes,
                0,
                padding as usize,
                &stream,
            )?;
        }
    }
    publish_bound(parts, batch, call, index, &write, true)
}

/// Decode this rank's audio units into its span of the track.
fn decode_audio(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    batch: &mut MediaBatch,
    index: usize,
    slot: usize,
    placement: &Placement<'_>,
) -> Result<()> {
    let call = &plan.calls[index];
    let range = decode_range(plan, call)?;
    let samples = parts.video.request(slot, call.request_key)?.audio_samples;
    if call.inputs.len() != 1 || call.outputs.len() != 1 {
        return Err(Error::Invalid(
            "media reconstruction requires one tensor input and output".into(),
        ));
    }
    let rank = parts.video.rank;
    let audio = parts
        .video
        .audio
        .as_ref()
        .ok_or_else(|| Error::Invalid("this rank holds no audio decoder".into()))?;
    let total = if audio.config.distribution.is_some() {
        (audio.config.ranks.len() * audio.config.units_per_rank.max(1)) as u32
    } else {
        1
    };
    let units = range.max_units.min(total).max(1);
    let run = audio.units(rank, 0, units)?;
    if run.is_empty() {
        return Ok(());
    }
    let spans = parts.video.model.audio_units(samples, units)?;
    let first = spans[run.start as usize].start;
    let last = spans[run.end as usize - 1].end;
    let owners = parts.owners();
    let stream = stream_of(batch)?;
    let latent = batch.inputs.read(
        owners.store,
        &call.inputs[0],
        call.call_id,
        batch.device,
        Some(&stream),
    )?;
    let write = bind_outputs(
        owners.store,
        &[OutputShape {
            product: call.outputs[0].clone(),
            shape: vec![samples, 2],
            region: Some(vec![first..last, 0..2]),
        }],
        batch.device,
        placement,
    )?
    .into_iter()
    .next()
    .ok_or_else(|| Error::Invariant("an audio span was not bound".into()))?;
    batch.calls[index].writes.push(write.clone());
    let audio = &parts
        .video
        .audio
        .as_ref()
        .ok_or_else(|| Error::Invalid("this rank holds no audio decoder".into()))?
        .runner;
    let written = audio.decode(
        latent.value(),
        samples,
        run.start,
        run.len() as u32,
        units,
        &write.view(),
    )?;
    if written != last - first {
        return Err(Error::Invalid(
            "audio units disagree with their sample span".into(),
        ));
    }
    publish_bound(parts, batch, call, index, &write, true)
}

/// Encode a request's conditions into denoiser rows: a round of visual
/// units, this rank's of them, or every audio track.
fn encode_conditions(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    batch: &mut MediaBatch,
    index: usize,
    slot: usize,
    placement: &Placement<'_>,
) -> Result<()> {
    let call = &plan.calls[index];
    if call.inputs.len() != 1 || call.outputs.len() != 1 {
        return Err(Error::Invalid(
            "condition encoding requires one input and one output".into(),
        ));
    }
    let name = parts.video.product_name(call, &call.outputs[0])?.to_owned();
    let conditions = Arc::clone(&parts.video.request(slot, call.request_key)?.conditions);
    if conditions.is_empty() {
        return Err(Error::Invalid(
            "condition encoding requires admitted conditions".into(),
        ));
    }
    let rank = parts.video.rank;
    let latent = parts
        .video
        .latent
        .as_ref()
        .ok_or_else(|| Error::Invalid("this rank holds no latent encoder".into()))?;
    let owners = parts.owners();
    let stream = stream_of(batch)?;
    let width = row_width(&call.outputs[0])?;
    match name.as_str() {
        "condition_video_latents" => {
            let range = decode_range(plan, call)?;
            // Every visual unit as (condition, unit, rows), in encoding order.
            let all: Vec<(usize, u32, u64)> = conditions
                .iter()
                .enumerate()
                .flat_map(|(index, condition)| {
                    condition
                        .latent_units
                        .iter()
                        .enumerate()
                        .map(move |(unit, &rows)| (index, unit as u32, u64::from(rows)))
                })
                .collect();
            let (cursor, count) = (range.cursor as usize, range.max_units as usize);
            if count == 0 || cursor + count > all.len() {
                return Err(Error::Invalid(
                    "condition encoding exceeds the admitted units".into(),
                ));
            }
            let round = &all[cursor..cursor + count];
            let run = latent.units(rank, 0, count as u32)?;
            if run.is_empty() {
                return Err(Error::Invalid(
                    "condition encoding assigns this rank no unit".into(),
                ));
            }
            let rows: u64 = round.iter().map(|unit| unit.2).sum();
            let first: u64 = round[..run.start as usize].iter().map(|unit| unit.2).sum();
            let last: u64 = round[..run.end as usize].iter().map(|unit| unit.2).sum();
            let pixels = batch.inputs.read(
                owners.store,
                &call.inputs[0],
                call.call_id,
                batch.device,
                Some(&stream),
            )?;
            let mut offsets = Vec::new();
            let mut cursor_pixels = 0u64;
            for condition in conditions.iter() {
                offsets.push(cursor_pixels);
                cursor_pixels += condition.pixel_bytes() / 3;
            }
            let units: Vec<[i64; 5]> = round[run.start as usize..run.end as usize]
                .iter()
                .map(|&(condition, unit, _)| {
                    let (frames, canvas) = conditions[condition].media.pixels().unwrap_or((
                        0,
                        Canvas {
                            width: 0,
                            height: 0,
                        },
                    ));
                    [
                        offsets[condition] as i64,
                        i64::from(frames),
                        i64::from(canvas.width),
                        i64::from(canvas.height),
                        i64::from(unit),
                    ]
                })
                .collect();
            let write = bind_outputs(
                owners.store,
                &[OutputShape {
                    product: call.outputs[0].clone(),
                    shape: vec![rows, width],
                    region: Some(vec![first..last, 0..width]),
                }],
                batch.device,
                placement,
            )?
            .into_iter()
            .next()
            .ok_or_else(|| Error::Invariant("condition rows were not bound".into()))?;
            batch.calls[index].writes.push(write.clone());
            let encoded = latent
                .runner
                .encode_video(pixels.value(), &units, &write.view())?;
            let planned: Vec<i64> = round[run.start as usize..run.end as usize]
                .iter()
                .map(|unit| unit.2 as i64)
                .collect();
            if encoded != planned {
                return Err(Error::Invalid(
                    "encoded condition units disagree with their planned rows".into(),
                ));
            }
            publish_bound(parts, batch, call, index, &write, false)?;
        }
        "condition_audio_latents" => {
            // The audio call is one unit, the component's first rank's.
            let run = latent.units(rank, 0, 1)?;
            if run.is_empty() {
                return Ok(());
            }
            let lengths: Vec<u64> = conditions
                .iter()
                .filter_map(|condition| condition.media.audio().map(|track| track.samples))
                .collect();
            let rows: u64 = conditions
                .iter()
                .map(|condition| u64::from(condition.audio_rows))
                .sum();
            let samples = batch.inputs.read(
                owners.store,
                &call.inputs[0],
                call.call_id,
                batch.device,
                Some(&stream),
            )?;
            let write = bind_outputs(
                owners.store,
                &[OutputShape {
                    product: call.outputs[0].clone(),
                    shape: vec![rows, width],
                    region: None,
                }],
                batch.device,
                placement,
            )?
            .into_iter()
            .next()
            .ok_or_else(|| Error::Invariant("condition rows were not bound".into()))?;
            batch.calls[index].writes.push(write.clone());
            let encoded = latent
                .runner
                .encode_audio(samples.value(), &lengths, &write.view())?;
            let planned: Vec<i64> = conditions
                .iter()
                .filter(|condition| condition.audio_rows > 0)
                .map(|condition| i64::from(condition.audio_rows))
                .collect();
            if encoded != planned {
                return Err(Error::Invalid(
                    "encoded audio tracks disagree with their planned rows".into(),
                ));
            }
            publish_bound(parts, batch, call, index, &write, false)?;
        }
        other => {
            return Err(Error::Invalid(format!(
                "latent encoding does not produce {other}"
            )));
        }
    }
    Ok(())
}

impl crate::executor::lifecycle::RequestStorage for VideoRuntime {
    fn admit_slots(&mut self, _slots: &[usize]) -> Result<()> {
        // A request's state is created when its admission is installed.
        Ok(())
    }

    fn release_buffers(&mut self, _buffers: &HashSet<BufferId>) -> Result<()> {
        Ok(())
    }

    fn release_requests(
        &mut self,
        _requests: &HashSet<RequestKey>,
        _retained: &HashSet<BufferId>,
    ) -> Result<()> {
        Ok(())
    }

    fn retirement_ready(
        &mut self,
        _buffers: &HashSet<BufferId>,
        requests: &HashSet<RequestKey>,
        _retained: &HashSet<BufferId>,
    ) -> Result<bool> {
        // A drawing task writes its slot's sources until it finishes.
        Ok(self.requests.values().all(|request| {
            !requests.contains(&request.key)
                || request
                    .preparation
                    .as_ref()
                    .is_none_or(|task| task.completion.done())
        }))
    }

    fn retire_slots(&mut self, slots: &[usize]) -> Result<()> {
        let mut released = Vec::new();
        for slot in slots {
            if let Some(request) = self.requests.remove(slot) {
                if request.layout.is_some() {
                    released.push(*slot);
                }
                self.retire(request)?;
            }
        }
        if let Some(denoiser) = &mut self.denoiser
            && !released.is_empty()
        {
            denoiser.runner.release_slots(&released)?;
        }
        Ok(())
    }

    fn reap(&mut self) -> Result<()> {
        self.outputs.reap();
        Ok(())
    }

    fn set_wake(&self, wake: Option<uniserve_ipc::Wake>) {
        if let Some(lane) = &self.noise {
            lane.set_wake(wake.clone());
        }
        if let Some(host) = &self.host {
            host.lane.set_wake(wake);
        }
    }

    fn close(&mut self, aborted: bool) -> Result<()> {
        let mut result = Ok(());
        for lane in self
            .noise
            .iter()
            .chain(self.host.as_ref().map(|host| &host.lane))
        {
            if aborted {
                lane.abort();
            } else {
                for error in lane.close() {
                    result = result.and(Err(Error::Invariant(format!(
                        "closing a host lane failed: {error}"
                    ))));
                }
            }
        }
        self.requests.clear();
        if let Some(text) = &mut self.text {
            result = result.and(text.runner.close(aborted));
        }
        if let Some(denoiser) = &mut self.denoiser {
            result = result.and(denoiser.runner.close(aborted));
        }
        if let Some(decoder) = &mut self.decoder {
            result = result.and(decoder.runner.close(aborted));
        }
        if let Some(audio) = &mut self.audio {
            result = result.and(audio.runner.close(aborted));
        }
        if let Some(latent) = &mut self.latent {
            result = result.and(latent.runner.close(aborted));
        }
        self.outputs.begin_close();
        self.outputs.finish_close();
        result
    }
}

impl super::batch::Trajectories for VideoDenoiser {
    fn validate(&self, updates: &[crate::LatentUpdate]) -> Result<()> {
        VideoDenoiser::validate(self, updates)
    }

    fn apply(&mut self, updates: &[crate::LatentUpdate]) {
        VideoDenoiser::apply(self, updates);
    }
}

/// What the video runtime is built from at startup.
pub(crate) struct RuntimeSetup<'a> {
    pub(crate) interface: &'a crate::model::interface::Interface,
    pub(crate) loaded: &'a Handle,
    pub(crate) model: VideoModel,
    pub(crate) capacity: crate::capacity::video::VideoCapacity,
    pub(crate) info: &'a WorkerInfo,
    pub(crate) config: &'a crate::config::WorkerConfig,
    pub(crate) components: &'a crate::model::registry::Components,
    pub(crate) lanes: &'a mut crate::worker::Lanes,
    pub(crate) graphs: &'a mut crate::runner::GraphStorage<String, Handle>,
    /// The attention backend the model loaded with.
    pub(crate) attention: String,
    /// The product arena of the rank's device, which encoder bindings name
    /// products in.
    pub(crate) products: Option<crate::storage::memory::StorageView>,
}

/// Graph storage keys of the video runtime's owners.
const DENOISER_GRAPHS: usize = 1000;
const DECODER_GRAPHS: usize = 1001;
const TEXT_GRAPHS: usize = 1002;

/// A stream forked for one component's calls, its workspace allocator, and
/// a borrowed view of it.
fn component_stream(
    lanes: &mut crate::worker::Lanes,
    kind: MediaCall,
    device: Device,
) -> Result<(Stream, tvm_ffi::Function)> {
    let Device::Cuda(index) = device else {
        return Err(Error::Unsupported(
            "video components compute on CUDA devices".into(),
        ));
    };
    let forked = lanes.fork_module(CallKind::Media(kind), uniserve_cuda::DeviceId(index))?;
    let raw = forked.stream()?.raw();
    // SAFETY: the lanes own the forked stream and close it after the runtime.
    let stream = unsafe { Stream::borrowed(uniserve_cuda::DeviceId(index), raw) }?;
    let workspaces = Arc::new(std::sync::Mutex::new(crate::storage::Workspaces::new(
        Some(index),
    )));
    Ok((stream, crate::runner::allocate::allocator(workspaces)))
}

impl VideoRuntime {
    /// Build the runtime of the video components placed on this rank.
    pub(crate) fn create(setup: RuntimeSetup<'_>) -> Result<Self> {
        let RuntimeSetup {
            interface,
            loaded,
            model,
            capacity,
            info,
            config,
            components,
            lanes,
            graphs,
            attention,
            products: arena,
        } = setup;
        let crate::capacity::video::VideoCapacity {
            media: plan,
            pages,
            envelope,
            outputs: capacity_outputs,
            overlap_bytes: _,
        } = capacity;
        let products: HashMap<String, Vec<String>> = info
            .components
            .iter()
            .map(|component| {
                (
                    component.name.clone(),
                    component
                        .outputs
                        .iter()
                        .map(|output| output.name.clone())
                        .collect(),
                )
            })
            .collect();
        let output_ranks: HashMap<String, usize> = info
            .components
            .iter()
            .filter_map(|component| {
                Some((component.name.clone(), *component.config.ranks.first()?))
            })
            .collect();
        let slots = info.request_slots;
        let capture = config.graph_policy != "off" && config.device.is_cuda();
        let create_pool = interface.function("graphs.pool")?;
        let pool = |graphs: &mut crate::runner::GraphStorage<String, Handle>,
                    key: usize,
                    owner: &str,
                    device: Device,
                    share: Option<usize>|
         -> Result<Option<Arc<crate::runner::GraphPool<Handle>>>> {
            let Device::Cuda(index) = device else {
                return Ok(None);
            };
            if !capture {
                return Ok(None);
            }
            let pools = crate::runner::graphs::reserve_pools(
                graphs,
                &create_pool,
                key,
                owner.to_owned(),
                &[index],
                share,
            )?;
            Ok(pools.into_iter().next())
        };
        let to_device = |device: crate::config::Device| match device {
            crate::config::Device::Cuda(index) => Device::Cuda(index),
            crate::config::Device::Cpu => Device::Cpu,
        };

        let mut text = None;
        let mut text_pool = None;
        let mut denoiser = None;
        let mut decoder = None;
        let mut audio = None;
        let mut latent = None;
        let mut host_components = BTreeMap::new();
        let conditioning = info
            .components
            .iter()
            .flat_map(|component| &component.outputs)
            .find(|output| output.name == "conditioning")
            .map(|output| -> Result<(u64, DType)> {
                let width = output
                    .shape_bound
                    .dims
                    .get(1)
                    .map(extent)
                    .ok_or_else(|| Error::Interface("text features are rows".into()))?;
                Ok((
                    width,
                    crate::storage::memory::parse_dtype(output.dtype.storage_name())?,
                ))
            })
            .transpose()?;
        for placed in components.placed() {
            let device = to_device(placed.device);
            let has = |capability| placed.has(capability);
            if placed.host() {
                host_components.insert(placed.name.clone(), placed.config.clone());
                continue;
            }
            if has(crate::model::components::Capability::TextEncoder) {
                // The media reader packs a request's vision blocks into the
                // rows of `vision_pixels`; a deployment without conditioned
                // tasks declares none.
                let vision_rows = capacity_outputs
                    .get("media_reader")
                    .and_then(|outputs| {
                        outputs.iter().find(|output| output.name == "vision_pixels")
                    })
                    .and_then(|output| output.shape_bound.dims.first())
                    .map_or(0, |dim| extent(dim) as usize);
                let path = placed
                    .calls
                    .iter()
                    .find(|call| call.is(crate::model::components::Capability::TextEncoder))
                    .map(|call| call.path.clone())
                    .unwrap_or_default();
                let products = arena.clone().ok_or_else(|| {
                    Error::Invariant("a text encoding rank has no product arena".into())
                })?;
                let (stream, _) = component_stream(lanes, MediaCall::TextEncoding, device)?;
                text_pool = pool(graphs, TEXT_GRAPHS, &placed.name, device, None)?;
                let runner = TextEncoding::new(TextSetup {
                    interface,
                    loaded,
                    component: &placed.name,
                    path: &path,
                    device: device_index(device)?,
                    stream,
                    attention: attention.clone(),
                    dtype: crate::runner::text::model_dtype(&config.model_dtype)?,
                    config: EncoderConfig {
                        // The text encoder derives its vision blocks'
                        // bound from the row budget.
                        max_images: 1,
                        row_budget: vision_rows.max(1),
                        text_capacities: plan
                            .text_capacities
                            .iter()
                            .map(|&capacity| capacity as usize)
                            .collect(),
                        max_text_tokens: plan.max_text_tokens as usize,
                    },
                    products,
                    vision: has(crate::model::components::Capability::PatchEncoder),
                })?;
                text = Some(Component {
                    name: placed.name.clone(),
                    config: placed.config.clone(),
                    device,
                    runner,
                });
            }
            if placed.name == model.denoiser {
                let envelope = envelope
                    .clone()
                    .ok_or_else(|| Error::Invariant("a denoising rank has no slot state".into()))?;
                let (stream, allocate) = component_stream(lanes, MediaCall::Denoising, device)?;
                let graph_pool = pool(graphs, DENOISER_GRAPHS, &placed.name, device, None)?;
                let conditioner = placed.calls.iter().any(|call| {
                    call.method == "encode"
                        && call.is(crate::model::components::Capability::Encoder)
                });
                let conditioning = match (conditioner, conditioning) {
                    (true, Some((width, dtype))) => {
                        Some((plan.text_capacities.clone(), width, dtype))
                    }
                    (true, None) => {
                        return Err(Error::Unsupported(
                            "the text conditioner has no text encoder features".into(),
                        ));
                    }
                    (false, _) => None,
                };
                let Device::Cuda(index) = device else {
                    return Err(Error::Unsupported(
                        "the video denoiser computes on CUDA".into(),
                    ));
                };
                let runner = VideoDenoiser::new(crate::runner::diffusion::video::DenoiserSetup {
                    bind: &interface.function("video.bind_denoiser")?,
                    loaded,
                    component: &placed.name,
                    stream,
                    device: index,
                    maximum: plan.maximum,
                    envelope,
                    pages,
                    slots,
                    conditioning,
                    conditioning_width: u64::from(model.facts.text_condition_width),
                    attention: attention.clone(),
                    allocate,
                    graph_pool,
                    transient_layouts: config.max_request_pool_size,
                })?;
                denoiser = Some(Component {
                    name: placed.name.clone(),
                    config: placed.config.clone(),
                    device,
                    runner,
                });
            }
            if has(crate::model::components::Capability::VideoDecoder) {
                let (stream, allocate) = component_stream(lanes, MediaCall::VideoDecoding, device)?;
                let graph_pool = pool(graphs, DECODER_GRAPHS, &placed.name, device, None)?;
                let mut window = (0, DType::F32);
                for segment in segments(&model, &plan)? {
                    let size = model.window(segment)?;
                    window = (window.0.max(size.0), size.1);
                }
                let mut overlap = (0, DType::F16);
                for &canvas in &plan.canvases {
                    let size = model.overlap(canvas)?;
                    overlap = (overlap.0.max(size.0), size.1);
                }
                let runner = VideoDecoding::new(
                    crate::runner::video::BindingSetup {
                        bind: &interface.function("video.bind_decoder")?,
                        loaded,
                        component: &placed.name,
                        stream,
                        device: device_index(device)?,
                        attention: attention.clone(),
                        allocate,
                    },
                    window,
                    overlap,
                    slots,
                    graph_pool,
                )?;
                decoder = Some(Component {
                    name: placed.name.clone(),
                    config: placed.config.clone(),
                    device,
                    runner,
                });
            }
            if has(crate::model::components::Capability::AudioDecoder) {
                let (stream, allocate) = component_stream(lanes, MediaCall::AudioDecoding, device)?;
                let runner = AudioDecoding::new(crate::runner::video::BindingSetup {
                    bind: &interface.function("video.bind_audio_decoder")?,
                    loaded,
                    component: &placed.name,
                    stream,
                    device: device_index(device)?,
                    attention: attention.clone(),
                    allocate,
                })?;
                audio = Some(Component {
                    name: placed.name.clone(),
                    config: placed.config.clone(),
                    device,
                    runner,
                });
            }
            if has(crate::model::components::Capability::VideoEncoder) {
                let (stream, allocate) =
                    component_stream(lanes, MediaCall::LatentEncoding, device)?;
                let runner = LatentEncoding::new(crate::runner::video::BindingSetup {
                    bind: &interface.function("video.bind_latent_encoder")?,
                    loaded,
                    component: &placed.name,
                    stream,
                    device: device_index(device)?,
                    attention: attention.clone(),
                    allocate,
                })?;
                latent = Some(Component {
                    name: placed.name.clone(),
                    config: placed.config.clone(),
                    device,
                    runner,
                });
            }
        }
        let host = if host_components.is_empty() {
            None
        } else {
            let reader = if host_components.contains_key("media_reader") {
                let functions = crate::host::reader::ReaderFunctions::new(interface, loaded)?;
                Some(Arc::new(ConditionReader::new(
                    functions,
                    config.ffmpeg.clone(),
                    model.facts.frame_rate,
                )))
            } else {
                None
            };
            let capacity =
                (info.queue_depth as usize * info.max_batch_calls.max(1) as usize).max(1);
            Some(HostMedia {
                reader,
                encode_video: interface.function("media.encode_video_unit")?,
                encode_audio: interface.function("media.encode_audio_track")?,
                mux: MuxFunctions::new(interface)?,
                lane: HostLane::new(capacity, 1, "uniserve-media")?,
                components: host_components,
            })
        };
        let noise = match &denoiser {
            Some(_) => Some(HostLane::new(
                config.max_request_pool_size.max(1),
                1,
                "uniserve-noise",
            )?),
            None => None,
        };
        let device = match config.device {
            crate::config::Device::Cuda(index) => Some(index),
            crate::config::Device::Cpu => None,
        };
        let leases = 2 * info.queue_depth as usize + 2;
        let words = info.max_batch_calls.max(1) as usize;
        Ok(Self {
            model,
            plan,
            rank: config.rank,
            requests: HashMap::new(),
            text,
            text_pool,
            denoiser,
            decoder,
            audio,
            latent,
            host,
            noise,
            outputs: OutputPool::new(device, leases, words),
            products,
            output_ranks,
        })
    }

    /// The bindings whose kernels the rank reports.
    pub(crate) fn kernels(&self) -> Result<Vec<(String, Vec<crate::ffi::Record>)>> {
        let mut functions = Vec::new();
        if let Some(denoiser) = &self.denoiser {
            functions.push((denoiser.name.clone(), denoiser.runner.kernels().clone()));
        }
        if let Some(decoder) = &self.decoder {
            functions.push((decoder.name.clone(), decoder.runner.kernels().clone()));
        }
        if let Some(audio) = &self.audio {
            functions.push((audio.name.clone(), audio.runner.kernels().clone()));
        }
        if let Some(latent) = &self.latent {
            functions.push((latent.name.clone(), latent.runner.kernels().clone()));
        }
        let mut kernels = Vec::new();
        if let Some(text) = &self.text {
            kernels.push((text.name.clone(), text.runner.kernels()?));
        }
        for (label, function) in functions {
            let value: tvm_ffi::Array<Any> = function.call_as(&[])?;
            let records = value
                .iter()
                .enumerate()
                .map(|(index, record)| {
                    crate::ffi::Record::new(format!("{label}.kernels[{index}]"), record)
                })
                .collect::<Result<Vec<_>>>()?;
            kernels.push((label, records));
        }
        Ok(kernels)
    }

    /// Prepare and warm every startup size: the denoiser's layouts (the
    /// largest first, so it sizes the shared workspace) and conditioner
    /// capacities, the decoder's segments and every round shape of its
    /// unit ring, and the audio decoder's durations. Runs on every rank of
    /// the group in the same order, because rounds exchange overlaps.
    pub(crate) fn prepare_startup(&mut self) -> Result<()> {
        let model = &self.model;
        let plan = &self.plan;
        if let Some(denoiser) = &mut self.denoiser {
            let runner = &mut denoiser.runner;
            let mut ids = Vec::new();
            for layout in std::iter::once(&plan.maximum)
                .chain(&plan.layouts)
                .chain(&plan.largest)
            {
                let id = runner.prepare_startup(model, *layout)?;
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
            runner.prepare_conditioner()?;
            // A placeholder request in slot one warms every layout before
            // any capture: kernels, plans and scratch are created by a
            // layout's first step.
            let prepare = runner.preparation();
            for &id in &ids {
                let key = runner.key(id)?;
                let size = VideoSize::plain(
                    key.frames,
                    key.canvas,
                    key.text_rows.min(plan.max_text_tokens),
                );
                prepare.call_values(&[
                    Any::from(1_i64),
                    Any::from(id as i64),
                    size.descriptor(),
                    size.conditions_descriptor(),
                    size.spans_descriptor(),
                    Any::from(0_i64),
                ])?;
                runner.warm(id, 1)?;
            }
            runner.warm_conditioner(false)?;
        }
        if let Some(decoder) = &mut self.decoder {
            let segments = segments(model, plan)?;
            decoder.runner.prepare(&segments, &plan.canvases)?;
            decoder.runner.warm(&segments)?;
            let width = (decoder.config.ranks.len() * decoder.config.units_per_rank.max(1)) as u32;
            for &canvas in &plan.canvases {
                for &frames in plan.frame_counts.iter().rev() {
                    let total = model.unit_frames(frames)?.len() as u32;
                    let mut cursor = 0;
                    while cursor < total {
                        let count = width.min(total - cursor);
                        let run = decoder.units(self.rank, cursor, count)?;
                        if !run.is_empty() {
                            decoder.runner.warm_unit(frames, canvas, run.start, count)?;
                        }
                        cursor += count;
                    }
                }
            }
            decoder.runner.stream().synchronize()?;
        }
        if let Some(audio) = &mut self.audio {
            let total = if audio.config.distribution.is_some() {
                (audio.config.ranks.len() * audio.config.units_per_rank.max(1)) as u32
            } else {
                1
            };
            let canvas = plan.canvases[0];
            for &frames in plan.frame_counts.iter().rev() {
                let samples = model.track_samples(frames)?;
                audio.runner.prepare(model.audio_latent_frames(samples)?)?;
            }
            for &frames in plan.frame_counts.iter().rev() {
                let samples = model.track_samples(frames)?;
                let layout =
                    model.layout_size(&VideoSize::plain(frames, canvas, plan.max_text_tokens))?;
                let shape = model.latent_shape("audio", &layout)?;
                let run = audio.units(self.rank, 0, total)?;
                if !run.is_empty() {
                    audio
                        .runner
                        .warm(samples, run.start, run.len() as u32, total, &shape)?;
                }
            }
            audio.runner.stream().synchronize()?;
        }
        Ok(())
    }

    /// Capture the denoiser's startup layouts and conditioner capacities and
    /// the decoder's segments, then refuse further captures.
    pub(crate) fn capture_startup(&mut self) -> Result<()> {
        if let Some(text) = &mut self.text {
            // Prompt capacities are warmed and captured largest first, the
            // same order on every rank of a tensor-parallel encoder.
            text.runner
                .capture(self.text_pool.as_ref(), &mut || Ok(()))?;
            text.runner.seal();
        }
        if let Some(denoiser) = &mut self.denoiser {
            let runner = &mut denoiser.runner;
            for layout in &self.plan.layouts {
                let id = runner.prepare_startup(&self.model, *layout)?;
                runner.capture(id)?;
            }
            runner.warm_conditioner(true)?;
            runner.stream().synchronize()?;
            runner.seal();
        }
        if let Some(decoder) = &mut self.decoder {
            let segments = segments(&self.model, &self.plan)?;
            decoder.runner.capture(&segments)?;
            decoder.runner.seal();
        }
        Ok(())
    }
}

/// The CUDA ordinal of a component's device.
fn device_index(device: Device) -> Result<i32> {
    match device {
        Device::Cuda(index) => Ok(index),
        Device::Cpu | Device::CudaHost => Err(Error::Unsupported(
            "video components compute on CUDA devices".into(),
        )),
    }
}

/// Every segment admitted videos decode to: canvases (in preparation order) by
/// frame counts (longest first) by units, without repeats.
pub(crate) fn segments(model: &VideoModel, plan: &MediaPlan) -> Result<Vec<Segment>> {
    let mut result = Vec::new();
    for &canvas in &plan.canvases {
        for &frames in plan.frame_counts.iter().rev() {
            for unit in 0..model.unit_frames(frames)?.len() as u32 {
                let segment = model.segment(frames, canvas, unit)?;
                if !result.contains(&segment) {
                    result.push(segment);
                }
            }
        }
    }
    Ok(result)
}

/// The warmup work of the video runtime.
pub(crate) struct VideoWarmup;

impl crate::warmup::WarmupHooks for VideoWarmup {
    fn prepare(
        &mut self,
        backend: &mut crate::executor::backend::WorkerBackend,
    ) -> std::result::Result<(), WorkerError> {
        if let Some(video) = backend.video_mut() {
            video
                .prepare_startup()
                .map_err(|error| WorkerError::classify(error, Some("warmup")))?;
        }
        Ok(())
    }

    fn capture(
        &mut self,
        backend: &mut crate::executor::backend::WorkerBackend,
    ) -> std::result::Result<(), WorkerError> {
        if let Some(video) = backend.video_mut() {
            video
                .capture_startup()
                .map_err(|error| WorkerError::classify(error, Some("capture")))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use uniserve_core::{ComponentDistribution, ParallelConfig};

    use super::*;

    fn config(ranks: &[usize], distribution: bool, units_per_rank: usize) -> ComponentConfig {
        ComponentConfig {
            ranks: ranks.to_vec(),
            parallel_config: ParallelConfig::default(),
            distribution: distribution.then_some(ComponentDistribution::TemporalUnits),
            units_per_rank,
        }
    }

    #[test]
    fn condition_latents_follow_their_rows_across_encoding_rounds() {
        // A keyframe of 4 rows, a reference video of 10 rows with 3 audio
        // rows and a voice of 2 audio rows; the encoder published the
        // visual rows in rounds of 6, 5 and 3 rows.
        let conditions = [(4, 0), (10, 3), (0, 2)];
        let latents = condition_spans(&conditions, &[6, 5, 3], Some(5)).unwrap();
        assert_eq!(
            latents,
            vec![
                vec![(0, 0, 4)],
                // The video continues through the second round into the third.
                vec![(0, 4, 2), (1, 0, 5), (2, 0, 3)],
                // The audio read follows the three visual rounds.
                vec![(3, 0, 3)],
                vec![(3, 3, 2)],
            ]
        );

        // Reads that hold other rows than the request planned are refused.
        assert!(condition_spans(&conditions, &[6, 5], Some(5)).is_err());
        assert!(condition_spans(&conditions, &[6, 5, 3], Some(4)).is_err());
        assert!(condition_spans(&conditions, &[6, 5, 3], None).is_err());
        // Without audio, every read is a visual round.
        assert_eq!(
            condition_spans(&[(4, 0)], &[4], None).unwrap(),
            vec![vec![(0, 0, 4)]]
        );
    }

    #[test]
    fn a_unit_ring_deals_each_round_from_its_first_member() {
        let ring = config(&[4, 5, 6, 7], true, 1);
        // A full round of four units starting at unit 8.
        assert_eq!(units(&ring, 5, 8, 4).unwrap(), 9..10);
        // A short round leaves the last members without a unit.
        assert!(units(&ring, 6, 12, 2).unwrap().is_empty());
        assert_eq!(units(&ring, 4, 12, 2).unwrap(), 12..13);
        // Members take consecutive runs of `units_per_rank`.
        assert_eq!(units(&config(&[0, 1], true, 2), 1, 0, 3).unwrap(), 2..3);
        // An undistributed component computes the whole round.
        assert_eq!(units(&config(&[0, 1], false, 1), 1, 3, 4).unwrap(), 3..7);
        assert!(units(&ring, 0, 0, 4).is_err());
    }
}

//! The video runtime: a rank's video requests and the order of their work.
//!
//! [`VideoRuntime`] holds the runners of the video components placed on a
//! rank: the text encoder with its vision tower, the denoiser with its
//! latent pool and layouts, the video and audio decoders, the condition
//! latent encoders (`crate::runner::{encoder, denoiser, decoder}`) and the
//! host media (`crate::runner::media`). It admits each video request the
//! rank serves when the engine starts it, keeps the request's numerical
//! size and layout while it lives, hands each media call of a batch to its
//! runner's forward, and warms and captures the runners at startup. Every
//! batch runs through a [`MediaBatch`](super::batch::MediaBatch): its inputs
//! are imported before it executes, its outputs are written and exported on
//! the call's stream, and its result is delivered once its device work and
//! host tasks completed.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;

use uniserve_core::{Canvas, ComponentConfig, VideoCondition};
use uniserve_cuda::Stream;
use uniserve_ipc::{
    Batch as BatchPlan, BufferAllocation, BufferId, Call, CallKind, MediaCall, NewRequest,
    RequestKey, TensorRef, WorkerInfo,
};

use super::batch::{MediaBatch, request_slots};
use crate::error::WorkerError;
use crate::executor::forward::placement::{
    OutputShape, Placement, ProductOwners, Reads, bind_completion, bind_outputs, export, skipped,
};
use crate::ffi::{DType, Device, Handle, Record};
use crate::host::mux::{MuxConfig, MuxSession};

use crate::request::RequestPool;
use crate::request::video::{MediaPlan, VideoRequest, VideoSize, condition_shapes, vision_spans};
use crate::runner::RunnerSetup;
use crate::runner::decoder::{DecodeCapacity, DecoderRunner, Segment, TrackUnits, UnitShape};
use crate::runner::denoiser::{Conditioning, DenoiserRunner, VideoModel};
use crate::runner::encoder::EncoderConfig;
use crate::runner::encoder::{EncoderRunner, TextSetup};
use crate::runner::media::MediaRunner;
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

/// A rank's video runtime.
///
/// FIXME: this composition of a rank's runners with its admitted requests,
/// their retirement and the choice of the runner serving a call is the
/// executor's and the request pool's; it stays here until
/// `executor/backend.rs` holds runners and requests instead of one runtime
/// per model family.
pub(crate) struct VideoRuntime {
    pub(crate) model: VideoModel,
    pub(crate) plan: MediaPlan,
    pub(crate) rank: usize,
    /// Admitted requests by slot.
    pub(crate) requests: HashMap<usize, VideoRequest>,
    pub(crate) text: Option<Component<EncoderRunner>>,
    /// The vision tower bound with the text encoder, for conditioned tasks.
    pub(crate) vision: Option<EncoderRunner>,
    /// The text encoder's graph pool, when graphs are captured.
    pub(crate) text_pool: Option<Arc<crate::cuda_graph::GraphPool<Handle>>>,
    pub(crate) denoiser: Option<Component<DenoiserRunner>>,
    pub(crate) decoder: Option<Component<DecoderRunner>>,
    /// The unit shapes startup warmed, which it then captures: each with
    /// its window's latent rows and its video's longest unit.
    decoder_shapes: Vec<(UnitShape, u64, u32)>,
    /// The track units startup warmed, which it then captures, each with
    /// its track's latent rows.
    audio_units: Vec<(TrackUnits, u64)>,
    pub(crate) audio: Option<Component<DecoderRunner>>,
    pub(crate) latent: Option<Component<EncoderRunner>>,
    pub(crate) host: Option<MediaRunner>,
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
    /// rank, its layout.
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
            seed: media.seed,
            mux: None,
        };
        if let Some(denoiser) = &mut self.denoiser {
            let admitted = self.plan.admit(&self.model, size)?;
            let id = denoiser.runner.acquire(&self.model, admitted.layout)?;
            request.layout = Some(id);
            request.admission = Some(admitted);
        }
        if let Some(previous) = self.requests.insert(slot, request) {
            self.retire(previous)?;
        }
        Ok(())
    }

    fn retire(&mut self, request: VideoRequest) -> Result<()> {
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
        match kind {
            MediaCall::TextEncoding | MediaCall::VisionEncoding => {
                let text = self
                    .text
                    .as_ref()
                    .ok_or_else(|| Error::Invalid("this rank holds no text encoder".into()))?;
                Ok((text.device, text.runner.stream().cloned()))
            }
            MediaCall::LatentPreparation | MediaCall::Denoising => {
                let denoiser = self
                    .denoiser
                    .as_ref()
                    .filter(|denoiser| denoiser.name == component);
                let denoiser = denoiser
                    .ok_or_else(|| Error::Invalid("this rank holds no video denoiser".into()))?;
                Ok((denoiser.device, Some(denoiser.runner.stream().clone())))
            }
            MediaCall::VideoDecoding => {
                let decoder = self
                    .decoder
                    .as_ref()
                    .ok_or_else(|| Error::Invalid("this rank holds no video decoder".into()))?;
                Ok((decoder.device, decoder.runner.stream().cloned()))
            }
            MediaCall::AudioDecoding => {
                let audio = self
                    .audio
                    .as_ref()
                    .ok_or_else(|| Error::Invalid("this rank holds no audio decoder".into()))?;
                Ok((audio.device, audio.runner.stream().cloned()))
            }
            MediaCall::LatentEncoding => {
                let latent = self
                    .latent
                    .as_ref()
                    .ok_or_else(|| Error::Invalid("this rank holds no latent encoder".into()))?;
                Ok((latent.device, latent.runner.stream().cloned()))
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
                crate::runner::encoder::encode_text(parts, plan, batch, index, slot, &placement)?
            }
            CallKind::Media(MediaCall::VisionEncoding) => {
                crate::runner::encoder::encode_vision(parts, plan, batch, index, slot, &placement)?
            }
            CallKind::Media(MediaCall::LatentPreparation) => {
                crate::runner::denoiser::prepare(parts, plan, batch, index, slot)?
            }
            CallKind::Media(MediaCall::Denoising) => {
                crate::runner::denoiser::denoise(parts, plan, batch, index, slot, &placement)?
            }
            CallKind::Media(MediaCall::VideoDecoding) => {
                crate::runner::decoder::decode_video(parts, plan, batch, index, slot, &placement)?
            }
            CallKind::Media(MediaCall::AudioDecoding) => {
                crate::runner::decoder::decode_audio(parts, plan, batch, index, slot, &placement)?
            }
            CallKind::Media(MediaCall::LatentEncoding) => {
                crate::runner::encoder::encode_conditions(
                    parts, plan, batch, index, slot, &placement,
                )?
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

pub(crate) fn latent_params<'a>(
    plan: &'a BatchPlan,
    call: &Call,
) -> Result<&'a uniserve_ipc::LatentParams> {
    plan.latent_params
        .iter()
        .find(|params| params.request_key == call.request_key && params.call_id == call.call_id)
        .ok_or_else(|| Error::Invalid("a video trajectory call has no latent pages".into()))
}

pub(crate) fn decode_range<'a>(
    plan: &'a BatchPlan,
    call: &Call,
) -> Result<&'a uniserve_ipc::DecodeRange> {
    plan.decode_ranges
        .iter()
        .find(|range| range.request_key == call.request_key && range.call_id == call.call_id)
        .ok_or_else(|| Error::Invalid("a media round has no unit range".into()))
}

/// The batch's stream.
pub(crate) fn stream_of(batch: &MediaBatch) -> Result<Stream> {
    batch
        .stream
        .clone()
        .ok_or_else(|| Error::Invariant("a device media call has no stream".into()))
}

/// The most elements a product holds along one dimension.
pub(crate) fn extent(dim: &uniserve_ipc::DimBound) -> u64 {
    match dim {
        uniserve_ipc::DimBound::Static(extent) | uniserve_ipc::DimBound::Device { max: extent } => {
            u64::from(*extent)
        }
    }
}

/// Rows' width of a two-dimensional product.
pub(crate) fn row_width(output: &TensorRef) -> Result<u64> {
    output
        .shape_bound
        .dims
        .get(1)
        .map(extent)
        .ok_or_else(|| Error::Invalid("the product is not rows".into()))
}

/// Bind a call's single output as rows of `shape` on every rank that
/// computes it.
pub(crate) fn bind_rows(
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
pub(crate) fn publish(
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
pub(crate) fn publish_bound(
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
        _requests: &HashSet<RequestKey>,
        _retained: &HashSet<BufferId>,
    ) -> Result<bool> {
        Ok(true)
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
        if let Some(host) = &self.host {
            host.lane.set_wake(wake);
        }
    }

    fn close(&mut self, aborted: bool) -> Result<()> {
        let mut result = Ok(());
        if let Some(lane) = self.host.as_ref().map(|host| &host.lane) {
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
        // The vision tower computes with the text encoder's binding, which
        // the text encoder releases.
        if let Some(vision) = &mut self.vision {
            result = result.and(vision.close(aborted));
        }
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

/// What the video runtime is built from at startup.
pub(crate) struct RuntimeSetup<'a> {
    pub(crate) interface: &'a crate::model::interface::Interface,
    pub(crate) loaded: &'a Handle,
    pub(crate) model: VideoModel,
    pub(crate) capacity: crate::capacity::video::VideoCapacity,
    pub(crate) info: &'a WorkerInfo,
    pub(crate) config: &'a crate::config::WorkerConfig,
    pub(crate) components: &'a crate::model::registry::Components,
    pub(crate) lanes: &'a mut crate::lane::Lanes,
    pub(crate) graphs: &'a mut crate::cuda_graph::GraphStorage<String, Handle>,
    /// The attention backend the model loaded with.
    pub(crate) attention: String,
}

/// Graph storage keys of the video runtime's owners.
const DENOISER_GRAPHS: usize = 1000;
const DECODER_GRAPHS: usize = 1001;
const TEXT_GRAPHS: usize = 1002;
const AUDIO_GRAPHS: usize = 1003;

/// A stream forked for one component's calls and its workspace allocator.
fn component_stream(
    lanes: &mut crate::lane::Lanes,
    kind: MediaCall,
    device: Device,
) -> Result<(Stream, tvm_ffi::Function)> {
    let Device::Cuda(index) = device else {
        return Err(Error::Unsupported(
            "video components compute on CUDA devices".into(),
        ));
    };
    let forked = lanes.fork_module(CallKind::Media(kind), uniserve_cuda::DeviceId(index))?;
    let stream = forked.stream()?.clone();
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
        let pool = |graphs: &mut crate::cuda_graph::GraphStorage<String, Handle>,
                    key: usize,
                    owner: &str,
                    device: Device,
                    share: Option<usize>|
         -> Result<Option<Arc<crate::cuda_graph::GraphPool<Handle>>>> {
            let Device::Cuda(index) = device else {
                return Ok(None);
            };
            if !capture {
                return Ok(None);
            }
            let pools = crate::cuda_graph::reserve_pools(
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
        let mut vision = None;
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
                let (stream, _) = component_stream(lanes, MediaCall::TextEncoding, device)?;
                text_pool = pool(graphs, TEXT_GRAPHS, &placed.name, device, None)?;
                let (runner, tower) = EncoderRunner::text(TextSetup {
                    interface,
                    loaded,
                    component: &placed.name,
                    path: &path,
                    device: device_index(device)?,
                    stream,
                    attention: attention.clone(),
                    dtype: crate::runner::causal_lm::model_dtype(&config.model_dtype)?,
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
                    vision: has(crate::model::components::Capability::PatchEncoder),
                })?;
                text = Some(Component {
                    name: placed.name.clone(),
                    config: placed.config.clone(),
                    device,
                    runner,
                });
                vision = tower;
            }
            if placed.name == model.denoiser {
                let envelope = envelope
                    .clone()
                    .ok_or_else(|| Error::Invariant("a denoising rank has no slot state".into()))?;
                let (stream, allocate) = component_stream(lanes, MediaCall::Denoising, device)?;
                let graph_pool = pool(graphs, DENOISER_GRAPHS, &placed.name, device, None)?;
                let conditioner = placed.calls.iter().find(|call| {
                    call.method == "encode"
                        && call.is(crate::model::components::Capability::Encoder)
                });
                let conditioning = match (conditioner, conditioning) {
                    (Some(call), Some((width, dtype))) => Some(Conditioning {
                        path: call.path.clone(),
                        capacities: plan.text_capacities.clone(),
                        width,
                        dtype,
                    }),
                    (Some(_), None) => {
                        return Err(Error::Unsupported(
                            "the text conditioner has no text encoder features".into(),
                        ));
                    }
                    (None, _) => None,
                };
                let Device::Cuda(index) = device else {
                    return Err(Error::Unsupported(
                        "the video denoiser computes on CUDA".into(),
                    ));
                };
                let runner = DenoiserRunner::new(crate::runner::denoiser::DenoiserSetup {
                    interface,
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
            // The module path of the placed call of a capability.
            let path = |capability: crate::model::components::Capability| -> Result<&str> {
                placed
                    .calls
                    .iter()
                    .find(|call| call.is(capability))
                    .map(|call| call.path.as_str())
                    .ok_or_else(|| {
                        Error::Invariant(format!(
                            "component {} calls no {}",
                            placed.name,
                            capability.tag()
                        ))
                    })
            };
            if has(crate::model::components::Capability::VideoDecoder) {
                let (stream, allocate) = component_stream(lanes, MediaCall::VideoDecoding, device)?;
                let graph_pool = pool(graphs, DECODER_GRAPHS, &placed.name, device, None)?;
                let mut capacity = DecodeCapacity {
                    latent: [0, 0],
                    frames: [0, 0, 0],
                    overlap: (0, DType::F16),
                };
                let longest = model
                    .unit_frames(plan.max_frames())?
                    .into_iter()
                    .max()
                    .unwrap_or(0);
                for &canvas in &plan.canvases {
                    let size = model.overlap(canvas)?;
                    capacity.overlap = (capacity.overlap.0.max(size.0), size.1);
                    let layout = model.layout_size(&VideoSize::plain(
                        plan.max_frames(),
                        canvas,
                        plan.max_text_tokens,
                    ))?;
                    let latent = model.latent_shape("video", &layout)?;
                    let [rows, width] = latent[..] else {
                        return Err(Error::Interface("a video latent is [rows, width]".into()));
                    };
                    capacity.latent = [capacity.latent[0].max(rows), width];
                    capacity.frames = [
                        u64::from(longest),
                        capacity.frames[1].max(u64::from(canvas.height)),
                        capacity.frames[2].max(u64::from(canvas.width)),
                    ];
                }
                let runner = DecoderRunner::video(
                    RunnerSetup {
                        interface,
                        loaded,
                        path: path(crate::model::components::Capability::VideoDecoder)?,
                        stream,
                        device: device_index(device)?,
                        attention: attention.clone(),
                        allocate,
                    },
                    capacity,
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
                let graph_pool = pool(graphs, AUDIO_GRAPHS, &placed.name, device, None)?;
                // A track's latent does not depend on the canvas.
                let layout = model.layout_size(&VideoSize::plain(
                    plan.max_frames(),
                    plan.canvases[0],
                    plan.max_text_tokens,
                ))?;
                let latent = model.latent_shape("audio", &layout)?;
                let [rows, width] = latent[..] else {
                    return Err(Error::Interface("an audio latent is [rows, width]".into()));
                };
                let runner = DecoderRunner::audio(
                    RunnerSetup {
                        interface,
                        loaded,
                        path: path(crate::model::components::Capability::AudioDecoder)?,
                        stream,
                        device: device_index(device)?,
                        attention: attention.clone(),
                        allocate,
                    },
                    [rows, width],
                    model.track_samples(plan.max_frames())?,
                    graph_pool,
                )?;
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
                let setup = |capability| -> Result<RunnerSetup<'_>> {
                    Ok(RunnerSetup {
                        interface,
                        loaded,
                        path: path(capability)?,
                        stream: stream.clone(),
                        device: device_index(device)?,
                        attention: attention.clone(),
                        allocate: allocate.clone(),
                    })
                };
                let runner = EncoderRunner::conditions(
                    setup(crate::model::components::Capability::VideoEncoder)?,
                    setup(crate::model::components::Capability::AudioEncoder)?,
                )?;
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
            let capacity =
                (info.queue_depth as usize * info.max_batch_calls.max(1) as usize).max(1);
            Some(MediaRunner::new(
                interface,
                loaded,
                host_components,
                config.ffmpeg.clone(),
                model.facts.frame_rate,
                capacity,
            )?)
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
            vision,
            text_pool,
            denoiser,
            decoder,
            decoder_shapes: Vec::new(),
            audio_units: Vec::new(),
            audio,
            latent,
            host,
            outputs: OutputPool::new(device, leases, words),
            products,
            output_ranks,
        })
    }

    /// The kernels each component's operators selected.
    pub(crate) fn kernels(&self) -> Result<Vec<(String, Vec<Record>)>> {
        let mut kernels = Vec::new();
        if let Some(text) = &self.text {
            kernels.push((text.name.clone(), text.runner.kernels(&text.name)?));
        }
        if let Some(denoiser) = &self.denoiser {
            kernels.push((
                denoiser.name.clone(),
                denoiser.runner.kernels(&denoiser.name)?,
            ));
        }
        if let Some(decoder) = &self.decoder {
            kernels.push((decoder.name.clone(), decoder.runner.kernels(&decoder.name)?));
        }
        if let Some(audio) = &self.audio {
            kernels.push((audio.name.clone(), audio.runner.kernels(&audio.name)?));
        }
        if let Some(latent) = &self.latent {
            kernels.push((latent.name.clone(), latent.runner.kernels(&latent.name)?));
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
            // A placeholder request in slot one warms every layout before
            // any capture: kernels, plans and scratch are created by a
            // layout's first step.
            for &id in &ids {
                let key = runner.key(id)?;
                let size = VideoSize::plain(
                    key.frames,
                    key.canvas,
                    key.text_rows.min(plan.max_text_tokens),
                );
                runner.prepare_latents(id, 1, &size, 0)?;
                runner.warm(id, 1)?;
            }
            runner.warm_conditioner(false)?;
        }
        if let Some(decoder) = &mut self.decoder {
            decoder.runner.prepare_segments(&segments(model, plan)?)?;
            let width = (decoder.config.ranks.len() * decoder.config.units_per_rank.max(1)) as u32;
            // One round of each shape warms the ring and is captured: a
            // round's shape is its canvas, whether it holds the video's
            // first and last units, and how many members hold a unit. Every
            // member derives the same rounds in the same order.
            let mut warmed = HashSet::new();
            for &canvas in &plan.canvases {
                for &frames in plan.frame_counts.iter().rev() {
                    let units = model.unit_frames(frames)?;
                    let total = units.len() as u32;
                    let longest = units.iter().copied().max().unwrap_or(0);
                    let mut cursor = 0;
                    while cursor < total {
                        let count = width.min(total - cursor);
                        let round = (canvas, cursor == 0, cursor + count == total, count);
                        let run = decoder.units(self.rank, cursor, count)?;
                        if warmed.insert(round) && !run.is_empty() {
                            let unit = run.start;
                            let shape = UnitShape {
                                segment: model.segment(frames, canvas, unit)?,
                                first: unit == 0,
                                last: unit + 1 == total,
                                count,
                            };
                            let rows = model.window_rows(frames, canvas, unit)?;
                            decoder.runner.warm_unit(shape, rows, longest)?;
                            self.decoder_shapes.push((shape, rows, longest));
                        }
                        cursor += count;
                    }
                }
            }
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
                audio
                    .runner
                    .prepare_track(model.audio_latent_frames(samples)?)?;
            }
            for &frames in plan.frame_counts.iter().rev() {
                let samples = model.track_samples(frames)?;
                let layout =
                    model.layout_size(&VideoSize::plain(frames, canvas, plan.max_text_tokens))?;
                let rows = model.latent_shape("audio", &layout)?[0];
                let run = audio.units(self.rank, 0, total)?;
                if !run.is_empty() {
                    let spans = model.audio_units(samples, total)?;
                    let units = TrackUnits {
                        samples,
                        latent_frames: model.audio_latent_frames(samples)?,
                        first: run.start,
                        count: run.len() as u32,
                        total,
                        span: spans[run.end as usize - 1].end - spans[run.start as usize].start,
                    };
                    audio.runner.warm_track(units, rows)?;
                    self.audio_units.push((units, rows));
                }
            }
        }
        Ok(())
    }

    /// Capture the text encoder's prompt capacities and the denoiser's
    /// startup layouts and conditioner capacities, then refuse further
    /// captures.
    pub(crate) fn capture_startup(&mut self) -> Result<()> {
        if let Some(text) = &mut self.text {
            // Prompt capacities are warmed and captured largest first, the
            // same order on every rank of a tensor-parallel encoder.
            text.runner
                .capture(self.text_pool.as_ref(), &mut || Ok(()))?;
            if let Some(stream) = text.runner.stream() {
                stream.synchronize()?;
            }
            text.runner.seal();
            if let Some(vision) = &mut self.vision {
                vision.seal();
            }
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
            for (shape, rows, longest) in std::mem::take(&mut self.decoder_shapes) {
                decoder.runner.capture_unit(shape, rows, longest)?;
            }
            if let Some(stream) = decoder.runner.stream() {
                stream.synchronize()?;
            }
            decoder.runner.seal();
        }
        if let Some(audio) = &mut self.audio {
            for (units, rows) in std::mem::take(&mut self.audio_units) {
                audio.runner.capture_track(units, rows)?;
            }
            if let Some(stream) = audio.runner.stream() {
                stream.synchronize()?;
            }
            audio.runner.seal();
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

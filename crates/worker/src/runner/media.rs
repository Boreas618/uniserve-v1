//! Host media: condition reading, video unit and audio encoding and MP4
//! assembly.
//!
//! Host media has no model. [`MediaRunner`] calls the functions of the
//! Python library's `uniserve.media` (and the model's patch packing)
//! by import path. Host calls run as tasks on the rank's host lane (see
//! [`crate::host::task`]); execution reserves their outputs, configures and
//! submits the tasks and commits the batch. Their outputs stay deferred
//! until the tasks finish, when the launched batch publishes them (the
//! decoded condition products, the encoded unit rows) or delivers the
//! finished MP4 before its result is sent
//! ([`crate::executor::batch::HostStage`]).
//!
//! The codecs and the muxer run on CPU ranks. The media reader runs on a
//! rank of the head's host, CPU or CUDA. A CPU rank's condition products are
//! host bytes, which encoders on its host map and a rank on another host
//! receives only over the rank channel, whose frames are bounded far below a
//! reference video's pixels. A CUDA rank's products lie in its device
//! storage: its task decodes into page-locked staging and copies it to them
//! on the reader's own stream, and encoder ranks on every host read them
//! through the device transport, so a deployment whose condition encoders
//! span hosts places the reader on a CUDA rank.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::{Arc, Mutex};

use uniserve_cuda::DeviceId;
use uniserve_ipc::ComponentConfig;
use uniserve_ipc::{Batch as BatchPlan, Call, DimBound, TensorRef, TransferTransport};

use crate::executor::products::{OutputShape, Placement, bind_outputs, supplied};
use crate::executor::video::{MediaBatch, Parts, Publication, product_name};
use crate::ffi::{Backing, Function, Handle, Interpreter, MemoryKind};
use crate::host::mux::{MuxFunctions, read_unit};
use crate::host::reader::{ConditionProducts, ConditionReader, ReaderFunctions};
use crate::host::task::{MediaTask, UnitFrames, Upload};
use crate::host::{HostLane, HostTask};
use crate::runner::component::units;
use crate::storage::PinnedPool;
use crate::storage::memory::StorageView;
use crate::stream::CUDAStream;
use crate::transfer::TransportKind;
use crate::{Error, ProductBuffer, Result};

/// The host media of a rank: the library functions its calls run and the
/// lane they run on.
pub(crate) struct MediaRunner {
    pub(crate) reader: Option<Arc<ConditionReader>>,
    /// On a CUDA rank holding the media reader, where its products are
    /// device storage: the page-locked staging the reader decodes into and
    /// the stream it copies the staging to the products on.
    staging: Option<DeviceStaging>,
    /// `uniserve.media.codec.encode_video_unit`.
    encode_video: Function,
    /// `uniserve.media.codec.encode_audio_track`.
    encode_audio: Function,
    pub(crate) mux: MuxFunctions,
    pub(crate) lane: HostLane<MediaTask>,
    /// Host component placements, by name.
    components: BTreeMap<String, ComponentConfig>,
}

/// A CUDA media reader's way to its device products.
struct DeviceStaging {
    device: DeviceId,
    pinned: Arc<PinnedPool>,
    stream: Arc<CUDAStream>,
}

impl MediaRunner {
    /// The runner of the host `components` placed on this rank, whose
    /// products lie on `device` (`None` for a CPU rank). A rank placing the
    /// media reader reads conditions with `ffmpeg` at the model's
    /// `frame_rate`; `capacity` bounds the lane's queued tasks.
    pub(crate) fn new(
        loaded: &Handle,
        components: BTreeMap<String, ComponentConfig>,
        ffmpeg: String,
        frame_rate: u32,
        capacity: usize,
        device: Option<DeviceId>,
    ) -> Result<Self> {
        let reader = if components.contains_key("media_reader") {
            let functions = ReaderFunctions::new(loaded)?;
            Some(Arc::new(ConditionReader::new(
                functions, ffmpeg, frame_rate,
            )))
        } else {
            None
        };
        let staging = match (&reader, device) {
            (Some(_), Some(device)) => Some(DeviceStaging {
                device,
                pinned: PinnedPool::new(Some(device)),
                stream: Arc::new(CUDAStream::create(device)?),
            }),
            _ => None,
        };
        Ok(Self {
            reader,
            staging,
            encode_video: Interpreter::import("uniserve.media.codec.encode_video_unit")?,
            encode_audio: Interpreter::import("uniserve.media.codec.encode_audio_track")?,
            mux: MuxFunctions::new()?,
            lane: HostLane::new(capacity, 1, "uniserve-media")?,
            components,
        })
    }

    /// Stop the host lane: abandon its tasks when `aborted`, otherwise
    /// finish them and report every failure.
    pub(crate) fn close(&self, aborted: bool) -> Result<()> {
        let mut result = Ok(());
        if aborted {
            self.lane.abort();
        } else {
            for error in self.lane.close() {
                result = result.and(Err(Error::Invariant(format!(
                    "closing a host lane failed: {error}"
                ))));
            }
        }
        result
    }
}

/// Reserve a host task, configure it with `action` and submit it.
fn submit(parts: &Parts<'_>, action: MediaTask) -> Result<Arc<HostTask<MediaTask>>> {
    let host = parts
        .media
        .ok_or_else(|| Error::Invalid("this rank holds no host media component".into()))?;
    let task = host.lane.reserve()?;
    if let Err(error) = task.configure(action).and_then(|()| task.submit()) {
        let _ = task.cancel(true);
        return Err(error);
    }
    Ok(task)
}

/// The static extent of an output's axis.
fn extent(product: &TensorRef, axis: usize) -> Result<u64> {
    match product.shape_bound.dims.get(axis) {
        Some(DimBound::Static(extent) | DimBound::Device { max: extent }) => Ok(u64::from(*extent)),
        None => Err(Error::Invalid("a media product lacks an axis".into())),
    }
}

/// Bind one host output at its exact shape and defer its write to a task.
fn bind_host(
    parts: &Parts<'_>,
    product: &TensorRef,
    shape: Vec<u64>,
    region: Option<Vec<Range<u64>>>,
    placement: &Placement<'_>,
) -> Result<ProductBuffer> {
    bind_deferred(parts, product, shape, region, placement, MemoryKind::Host)
}

/// Bind one output in `memory` at its exact shape and defer its write to a
/// task.
fn bind_deferred(
    parts: &Parts<'_>,
    product: &TensorRef,
    shape: Vec<u64>,
    region: Option<Vec<Range<u64>>>,
    placement: &Placement<'_>,
    memory: MemoryKind,
) -> Result<ProductBuffer> {
    let owners = parts.owners();
    let write = bind_outputs(
        owners.store,
        &[OutputShape {
            product: product.clone(),
            shape,
            region,
        }],
        memory,
        placement,
    )?
    .into_iter()
    .next()
    .ok_or_else(|| Error::Invariant("a host product was not bound".into()))?;
    owners.store.defer_write(&write)?;
    Ok(write)
}

/// Decode a conditioned request's media into its condition products.
pub(crate) fn read_media(
    parts: &mut Parts<'_>,
    batch: &mut MediaBatch,
    index: usize,
    call: &Call,
    slot: usize,
    placement: &Placement<'_>,
) -> Result<()> {
    let conditions = Arc::clone(&parts.video.request(slot, call.request_key)?.conditions);
    if conditions.is_empty() {
        return Err(Error::Invalid(
            "media reading requires admitted conditions".into(),
        ));
    }
    let reader = parts
        .media
        .and_then(|host| host.reader.clone())
        .ok_or_else(|| Error::Invalid("media reading requires the condition encoders".into()))?;
    let staging = parts.media.and_then(|host| host.staging.as_ref());
    let memory = staging.map_or(MemoryKind::Host, |staging| MemoryKind::Cuda(staging.device));
    let mut products = ConditionProducts {
        pixels: None,
        samples: None,
        patches: None,
    };
    let mut written = Vec::new();
    let mut copies = Vec::new();
    for output in &call.outputs {
        let name = product_name(parts.info, call, output)?.to_owned();
        let (rows, width): (u64, u64) = match name.as_str() {
            "condition_pixels" => (
                conditions
                    .iter()
                    .map(|condition| condition.pixel_bytes() / 3)
                    .sum(),
                3,
            ),
            "condition_samples" => (
                conditions
                    .iter()
                    .filter_map(|condition| condition.media.audio().map(|track| track.samples))
                    .sum(),
                2,
            ),
            "vision_pixels" => (
                conditions
                    .iter()
                    .filter_map(|condition| {
                        condition
                            .vision
                            .as_ref()
                            .map(|vision| vision.grid.patches())
                    })
                    .sum(),
                extent(output, 1)?,
            ),
            other => {
                return Err(Error::Invalid(format!(
                    "media reading does not produce {other}"
                )));
            }
        };
        let write = bind_deferred(parts, output, vec![rows, width], None, placement, memory)?;
        let mut view = write.view();
        if let Some(staging) = staging {
            // The reader decodes into page-locked staging of the product's
            // shape, which the task copies to the product.
            let block = staging.pinned.lease(view.nbytes() as usize)?;
            let host =
                StorageView::contiguous(block as Arc<dyn Backing>, 0, view.dtype(), view.shape())?;
            copies.push((host.clone(), view));
            view = host;
        }
        match name.as_str() {
            "condition_pixels" => products.pixels = Some(view),
            "condition_samples" => products.samples = Some(view),
            _ => products.patches = Some(view),
        }
        written.push((output.clone(), write));
    }
    let stream = staging.map(|staging| Arc::clone(&staging.stream));
    let upload = stream.clone().map(|stream| Upload { stream, copies });
    let task = submit(
        parts,
        MediaTask::Read {
            reader,
            conditions,
            products,
            upload,
        },
    )?;
    let state = &mut batch.batch.calls[index].forward;
    state.tasks.push(task);
    state.publication = Some(Publication::Products { written, stream });
    Ok(())
}

/// The decoded frames of one unit, read in place from a shared-memory
/// export on this host.
fn borrow_unit(
    parts: &Parts<'_>,
    plan: &BatchPlan,
    product: &TensorRef,
    row: u64,
) -> Result<UnitFrames> {
    let export = supplied(plan, product)
        .ok_or_else(|| Error::Invalid("a borrowed media input has no export".into()))?;
    let tensor = crate::executor::products::tensor(export)?;
    let transport = parts
        .storage
        .transports
        .get(TransportKind::PosixShm)
        .ok_or_else(|| Error::Invalid("video encoding requires local shared storage".into()))?;
    let location = tensor
        .locations
        .iter()
        .find(|location| {
            matches!(location.transport, TransferTransport::PosixShm { .. })
                && location.source.host == parts.info.endpoint.host
                && location.offset[0] <= row
                && row < location.offset[0] + location.shape[0]
        })
        .ok_or_else(|| Error::Invalid("decoded media unit has no local shared storage".into()))?;
    let region: Vec<Range<u64>> = std::iter::once(row..row + 1)
        .chain(tensor.shape[1..].iter().map(|&extent| 0..extent))
        .collect();
    let read = transport.borrow(location, Some(&region))?;
    Ok(UnitFrames::Borrowed(Mutex::new(read)))
}

/// Encode this rank's media units of a round into its rows of the round's
/// encoded product.
pub(crate) fn encode_units(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    batch: &mut MediaBatch,
    index: usize,
    call: &Call,
    slot: usize,
    placement: &Placement<'_>,
) -> Result<()> {
    let range = plan
        .decode_ranges
        .iter()
        .find(|range| range.request_key == call.request_key && range.call_id == call.call_id)
        .ok_or_else(|| Error::Invalid("video encoding has no decoder interval".into()))?;
    if call.inputs.len() != 1 || call.outputs.len() != 1 {
        return Err(Error::Invalid(
            "video encoding requires an input round and one output".into(),
        ));
    }
    let (canvas, frames_of) = {
        let request = parts.video.request(slot, call.request_key)?;
        (request.canvas, Arc::clone(&request.units))
    };
    let config = parts
        .media
        .and_then(|host| host.components.get(&call.component))
        .cloned()
        .ok_or_else(|| Error::Invalid("video encoding has no codec component".into()))?;
    let run = units(&config, &range.ranks, parts.rank, 0, range.max_units)?;
    if run.is_empty() {
        return Ok(());
    }
    let row_bytes = extent(&call.outputs[0], 1)?;
    let write = bind_host(
        parts,
        &call.outputs[0],
        vec![u64::from(range.max_units), row_bytes],
        Some(vec![u64::from(run.start)..u64::from(run.end), 0..row_bytes]),
        placement,
    )?;
    let rows = write.view();
    let imported = if batch.borrowed(&call.inputs[0]) {
        None
    } else {
        let store = parts.owners().store;
        Some(batch.read(store, &call.inputs[0], call.call_id, None)?)
    };
    let frame_rate = parts.video.model.facts.frame_rate;
    let function = parts
        .media
        .map(|host| host.encode_video.clone())
        .ok_or_else(|| Error::Invalid("this rank holds no video codec".into()))?;
    let mut tasks = Vec::new();
    let mut local = Vec::new();
    for position in run.clone() {
        let unit = range.cursor + position;
        let frames = *frames_of.get(unit as usize).ok_or_else(|| {
            Error::Invalid("video encoding names a unit outside its video".into())
        })?;
        let source = match &imported {
            Some(read) => UnitFrames::Host(read.value().select(0, i64::from(position))?),
            None => borrow_unit(parts, plan, &call.inputs[0], u64::from(position))?,
        };
        let row_index = i64::from(position - run.start);
        let row = rows.select(0, row_index)?;
        let task = submit(
            parts,
            MediaTask::EncodeUnit {
                function: function.clone(),
                frames: source,
                shape: (frames, canvas.height, canvas.width),
                frame_rate,
                row,
            },
        );
        match task {
            Ok(task) => tasks.push(task),
            Err(error) => {
                for task in &tasks {
                    let _ = task.cancel(false);
                }
                parts
                    .owners()
                    .store
                    .abandon_writes(std::slice::from_ref(&write))?;
                return Err(error);
            }
        }
        local.push(row_index as u64);
    }
    let state = &mut batch.batch.calls[index].forward;
    state.tasks = tasks;
    state.publication = Some(Publication::Encoded {
        product: call.outputs[0].clone(),
        write,
        rows: local,
    });
    Ok(())
}

/// Encode the request's soundtrack into its MP4 session.
pub(crate) fn encode_audio(
    parts: &mut Parts<'_>,
    batch: &mut MediaBatch,
    index: usize,
    call: &Call,
    slot: usize,
) -> Result<()> {
    if call.inputs.len() != 1 {
        return Err(Error::Invalid(
            "audio encoding requires one PCM input".into(),
        ));
    }
    let store = parts.owners().store;
    let pcm = batch.read(store, &call.inputs[0], call.call_id, None)?;
    let session = parts
        .video
        .mux_session(slot, call.request_key, parts.media)?;
    let samples = session.config().audio_samples();
    let rate = parts.video.model.facts.sample_rate;
    let function = parts
        .media
        .map(|host| host.encode_audio.clone())
        .ok_or_else(|| Error::Invalid("this rank holds no muxer".into()))?;
    let task = submit(
        parts,
        MediaTask::EncodeAudio {
            function,
            pcm: pcm.value().clone(),
            samples,
            rate,
            session,
        },
    )?;
    batch.batch.calls[index].forward.tasks.push(task);
    Ok(())
}

/// The framed unit payloads of one encoded round, in row order, read from
/// the exports that reach this rank.
fn read_rows(parts: &Parts<'_>, plan: &BatchPlan, product: &TensorRef) -> Result<Vec<Vec<u8>>> {
    let export = supplied(plan, product).ok_or_else(|| {
        Error::Invalid("artifact assembly input has no transfer locations".into())
    })?;
    let tensor = crate::executor::products::tensor(export)?;
    if tensor.shape.len() != 2
        || tensor
            .locations
            .iter()
            .any(|location| location.dtype != "uint8")
    {
        return Err(Error::Invalid("encoded units require byte rows".into()));
    }
    let mut rows: Vec<Option<Vec<u8>>> = vec![None; tensor.shape[0] as usize];
    for location in &tensor.locations {
        if location.offset.get(1).copied().unwrap_or(0) != 0 {
            return Err(Error::Invalid(
                "encoded unit location lacks its length prefix".into(),
            ));
        }
        let first = location.offset[0] as usize;
        let count = location.shape[0] as usize;
        let width = location.shape.get(1).copied().unwrap_or(0) as usize;
        if first + count > rows.len() || rows[first..first + count].iter().all(Option::is_some) {
            continue;
        }
        let bytes: Vec<u8> = match &location.transport {
            TransferTransport::Channel { payload, .. } => payload.clone(),
            TransferTransport::PosixShm { .. }
                if location.source.host == parts.info.endpoint.host =>
            {
                let Some(transport) = parts.storage.transports.get(TransportKind::PosixShm) else {
                    continue;
                };
                let mut read = transport.borrow(location, None)?;
                let mapping = read.mapping()?;
                // SAFETY: the claimed range is mapped and ready until the
                // read is released below.
                let payload = unsafe {
                    std::slice::from_raw_parts(
                        (mapping.address() + read.offset()) as *const u8,
                        read.nbytes(),
                    )
                }
                .to_vec();
                read.release();
                payload
            }
            _ => continue,
        };
        if width == 0 || bytes.len() != count * width {
            return Err(Error::Invalid(
                "encoded unit rows disagree with their location".into(),
            ));
        }
        for (offset, row) in bytes.chunks_exact(width).enumerate() {
            rows[first + offset] = Some(read_unit(row)?.to_vec());
        }
    }
    rows.into_iter()
        .map(|row| {
            row.ok_or_else(|| {
                Error::Invalid("artifact assembly requires every encoded media unit".into())
            })
        })
        .collect()
}

/// Append a request's encoded rounds to its MP4, or finish it.
pub(crate) fn mux(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    batch: &mut MediaBatch,
    index: usize,
    call: &Call,
    slot: usize,
) -> Result<()> {
    let session = parts
        .video
        .mux_session(slot, call.request_key, parts.media)?;
    let (action, publication) = if call.inputs.is_empty() {
        (MediaTask::Finalize { session }, Some(Publication::Media))
    } else {
        let mut units = Vec::new();
        for input in &call.inputs {
            units.extend(read_rows(parts, plan, input)?);
        }
        (MediaTask::Append { session, units }, None)
    };
    let task = submit(parts, action)?;
    let state = &mut batch.batch.calls[index].forward;
    state.tasks.push(task);
    state.publication = publication;
    Ok(())
}

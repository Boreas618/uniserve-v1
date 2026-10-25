//! Startup of a video model's rank: the executor and what the worker keeps
//! beside it.
//!
//! [`executor`] constructs the executor once the model is loaded and reads
//! its manifest, which capacity planning sizes the rank by. [`start`] then
//! allocates what the plan grants and the worker owns (the trajectory
//! ledger's latent banks, one stream per component, the graph pools'
//! device memory and the output leases) and binds the executor to it, which
//! constructs its runners. [`VideoWarmup`] prepares, warms and captures them
//! before the rank reports.

use std::collections::HashMap;
use std::sync::Arc;

use ipc::{MediaCall, WorkerInfo};

use crate::capacity::video::{VideoCapacity, VideoManifest};
use crate::config::placement::{Capability, Components, Declarations};
use crate::config::{Device, WorkerConfig};
use crate::cuda_graph::{GraphStorage, reserve_pools};
use crate::error::WorkerError;
use crate::executor::backend::WorkerBackend;
use crate::executor::python::PythonExecutor;
use crate::executor::video::{LatentLedger, VideoRuntime};
use crate::ffi::{Handle, Interpreter};
use crate::runner::scratch::Scratches;
use crate::stream::{CUDAStream, Streams};
use crate::warmup::WarmupHooks;
use crate::{Error, LatentPool, LatentStorage, OutputPool, Result};

/// Graph storage keys of a video model's components.
const DENOISER_GRAPHS: usize = 1000;
const DECODER_GRAPHS: usize = 1001;
const TEXT_GRAPHS: usize = 1002;
const AUDIO_GRAPHS: usize = 1003;
const LATENT_GRAPHS: usize = 1004;

/// The capabilities whose components capture graphs, each with its graph
/// storage key and the call kind its stream is named by; the kinds after it
/// compute on the same stream.
const CAPTURING: [(Capability, usize, MediaCall, &[MediaCall]); 5] = [
    (
        Capability::TextEncoder,
        TEXT_GRAPHS,
        MediaCall::TextEncoding,
        &[MediaCall::VisionEncoding],
    ),
    (
        Capability::VideoEncoder,
        LATENT_GRAPHS,
        MediaCall::LatentEncoding,
        &[],
    ),
    (
        Capability::VideoDenoiser,
        DENOISER_GRAPHS,
        MediaCall::Denoising,
        &[MediaCall::LatentPreparation],
    ),
    (
        Capability::VideoDecoder,
        DECODER_GRAPHS,
        MediaCall::VideoDecoding,
        &[],
    ),
    (
        Capability::AudioDecoder,
        AUDIO_GRAPHS,
        MediaCall::AudioDecoding,
        &[],
    ),
];

/// The video denoiser component a deployment places: the unique declared
/// video denoiser among `deployment` components (or the only one declared).
pub(crate) fn placed_denoiser(
    declarations: &Declarations,
    deployment: &[String],
) -> Result<String> {
    let declared: Vec<&str> = declarations
        .iter()
        .filter(|(_, calls)| calls.iter().any(|call| call.is(Capability::VideoDenoiser)))
        .map(|(name, _)| name)
        .collect();
    if let [only] = declared[..] {
        return Ok(only.to_owned());
    }
    let placed: Vec<&str> = declared
        .iter()
        .copied()
        .filter(|name| deployment.iter().any(|component| component == name))
        .collect();
    match placed[..] {
        [one] => Ok(one.to_owned()),
        _ => Err(Error::Unsupported(format!(
            "a deployment places exactly one of the model's video denoisers {declared:?}; it places {}",
            placed.len()
        ))),
    }
}

/// Construct the executor of a video model on this rank's deployment and
/// read its manifest.
pub(crate) fn executor(
    loaded: &Handle,
    config: &WorkerConfig,
    components: &Components,
    denoiser: &str,
    attention: &str,
) -> Result<(PythonExecutor, VideoManifest)> {
    let graphs = config.graph_policy != "off" && config.device.is_cuda();
    let placed: Vec<serde_json::Value> = components
        .placed()
        .iter()
        .map(|placed| {
            let calls: Vec<serde_json::Value> = placed
                .calls
                .iter()
                .flat_map(|call| {
                    call.capabilities.iter().map(move |capability| {
                        serde_json::json!({"capability": capability.tag(), "path": call.path})
                    })
                })
                .collect();
            let kinds: Vec<&str> = placed.kinds.iter().map(|kind| kind.as_str()).collect();
            serde_json::json!({
                "name": placed.name,
                "host": placed.host(),
                "calls": calls,
                "kinds": kinds,
                "ranks": placed.config.ranks,
                "distributed": placed.config.distribution.is_some(),
                "units_per_rank": placed.config.units_per_rank,
            })
        })
        .collect();
    let settings = serde_json::json!({
        "rank": config.rank,
        "device": config.device.cuda().map(|device| format!("cuda:{}", device.0)),
        "graphs": graphs,
        "attention": attention,
        "model_dtype": config.model_dtype,
        "max_model_len": config.max_model_len,
        "components": placed,
        "video": {
            "denoiser": denoiser,
            "resolutions": config.video_resolutions,
            "aspect_ratios": config.video_aspect_ratios,
            "max_seconds": config.max_video_seconds,
            "min_seconds": config.min_video_seconds,
            "text_capacities": config.video_text_capacities,
            "max_condition_rows": config.max_condition_rows,
        },
    });
    let executor = PythonExecutor::new(loaded, &settings.to_string())?;
    let manifest: serde_json::Value = serde_json::from_str(&executor.manifest()?)
        .map_err(|error| Error::Interface(format!("the executor's manifest: {error}")))?;
    let manifest: VideoManifest = serde_json::from_value(manifest["video"].clone())
        .map_err(|error| Error::Interface(format!("the executor's video manifest: {error}")))?;
    Ok((executor, manifest))
}

/// What [`start`] binds the executor with.
pub(crate) struct VideoStart<'a> {
    pub(crate) executor: PythonExecutor,
    pub(crate) capacity: &'a VideoCapacity,
    pub(crate) info: &'a WorkerInfo,
    pub(crate) config: &'a WorkerConfig,
    pub(crate) components: &'a Components,
    pub(crate) streams: &'a mut Streams,
    pub(crate) graphs: &'a mut GraphStorage,
    /// The shared work areas of the rank's streams.
    pub(crate) scratches: &'a mut Scratches,
    /// Unresolved batches the rank's launch admits at once.
    pub(crate) queue_depth: usize,
    /// Where the latent banks are taken.
    pub(crate) memory: &'a mut dyn crate::memory::ExclusiveMemory,
}

/// Allocate what the plan grants a video rank and bind the executor to it:
/// the executor constructs its runners.
pub(crate) fn start(start: VideoStart<'_>) -> Result<(VideoRuntime, OutputPool)> {
    let VideoStart {
        mut executor,
        capacity,
        info,
        config,
        components,
        streams,
        graphs,
        scratches,
        queue_depth,
        memory,
    } = start;
    let slots = info.request_slots as usize;
    let device = config.device.cuda();
    let capture = config.graph_policy != "off" && config.device.is_cuda();
    let create = Interpreter::import("uniserve.cuda_graph.pool")?;

    // One stream per component the rank holds, forked from the lane's; its
    // call kinds compute on it. The contexts of a
    // stream share its scratch.
    let mut kinds: HashMap<MediaCall, (Device, Option<cuda::Stream>)> = HashMap::new();
    let mut streams_json = serde_json::Map::new();
    let mut lane_scratches = HashMap::new();
    let mut pool_names = Vec::new();
    let mut pools = Vec::new();
    let mut ledger = None;
    for placed in components.placed() {
        if placed.host() {
            continue;
        }
        for &(capability, key, kind, others) in &CAPTURING {
            if !placed.has(capability) {
                continue;
            }
            let Some(cuda) = device else {
                return Err(Error::Unsupported(
                    "video components compute on CUDA devices".into(),
                ));
            };
            let stream = streams.fork_module(cuda)?.stream()?.clone();
            for &served in std::iter::once(&kind).chain(others) {
                kinds.insert(served, (Device::Cuda(cuda), Some(stream.clone())));
            }
            streams_json.insert(
                kind.as_str().to_owned(),
                serde_json::json!([stream.raw() as i64, cuda.0]),
            );
            lane_scratches.insert(
                kind.as_str().to_owned(),
                scratches.get(stream.raw(), Some(cuda))?,
            );
            if capture {
                let pool = reserve_pools(graphs, &create, key, &[cuda], None)?
                    .into_iter()
                    .next()
                    .ok_or_else(|| Error::Invariant("a graph pool was not reserved".into()))?;
                pool_names.push(capability.tag());
                pools.push(pool.value.clone());
            }
            if capability == Capability::VideoDenoiser {
                let plan = capacity.latent(slots);
                let storage = LatentStorage::allocate(
                    &plan,
                    Some(cuda),
                    Some(&stream),
                    "video.latent_banks",
                    memory,
                )?;
                let pool =
                    LatentPool::new(plan.request_pool_size, plan.num_pages, plan.page_units)?;
                let pages = capacity.manifest.pages;
                ledger = Some(LatentLedger::new(
                    pool,
                    storage,
                    pages.page_units,
                    pages.pages,
                ));
            }
        }
    }
    // Device products host tasks wrote are fenced on a stream of the rank's
    // device, once the tasks waited for their copies.
    let host_stream = match device {
        Some(cuda) if components.holds_host() => Some(Arc::new(CUDAStream::create(cuda)?)),
        _ => None,
    };
    let banks = ledger.as_ref().map(LatentLedger::banks).transpose()?;
    // The scratches follow the pools in the order the streams serialize.
    let lane_scratches = streams_json
        .keys()
        .map(|kind| lane_scratches.remove(kind))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| Error::Invariant("a lane has no scratch".into()))?;
    let memory_json = serde_json::json!({
        "slots": slots,
        "streams": streams_json,
        "pools": pool_names,
        "ffmpeg": config.ffmpeg,
        // A conditioned request whose layout startup did not prepare
        // evaluates in a transient one; the denoiser keeps one per slot.
        "transient_layouts": config.max_request_pool_size,
    });
    executor.bind(memory_json, pools, lane_scratches, banks)?;
    let leases = 2 * queue_depth + 2;
    let words = info.max_batch_calls.max(1) as usize;
    let outputs = OutputPool::new(device, leases, words)?;
    Ok((
        VideoRuntime::new(executor, ledger, kinds, host_stream),
        outputs,
    ))
}

/// The warmup of a video model's rank: the executor prepares, warms and
/// captures every runner, in the order every rank shares.
pub(crate) struct VideoWarmup;

impl WarmupHooks for VideoWarmup {
    fn prepare(&mut self, backend: &mut WorkerBackend) -> std::result::Result<(), WorkerError> {
        let video = backend.video().ok_or_else(|| {
            WorkerError::classify(
                Error::State("a video warmup has no executor"),
                Some("warmup"),
            )
        })?;
        video
            .executor
            .warmup()
            .map_err(|error| WorkerError::classify(error, Some("warmup")))
    }
}

//! Startup of the rank's executor and its computations: the executor's settings
//! and construction once the model is loaded, what its manifest states, the
//! storage the worker binds its computations to, and their warmup.
//!
//! One executor serves a rank ([`construct`]). Its settings carry a section per
//! role the rank serves: `video` for a video model, `image` for an image
//! model's computations (`crate::worker::image`), `pages` for the computations
//! over the prefix cache (the batch bounds and graph buckets, the cache groups
//! at the resolved block size, the block diffusion a generating rank serves)
//! and `experts` for its member of an expert union. Its manifest ([`Manifest`])
//! states each computation (its component and call kinds, whether it prepares
//! and commits, the resources it uses and its output words) and what capacity
//! planning reads: the device bytes the computations over the prefix cache
//! allocate beside its pool, and a video model's per-request quantities.
//! [`bind`] allocates what the plan grants and binds every computation to it;
//! [`warm`] runs the warmup the executor lays out in phases, around the prefix
//! cache's scratch tables the worker owns.

use std::collections::BTreeMap;
use std::sync::Arc;

use cuda::{DeviceId, Stream};
use ipc::CallKind;
use serde::Deserialize;

use crate::capacity::video::{VideoCapacity, VideoManifest};
use crate::config::placement::{Capability, Components, Declarations};
use crate::config::{WorkerConfig, WorkerLaunch};
use crate::cuda_graph::{GraphStorage, reserve_pools};
use crate::error::WorkerError;
use crate::executor::backend::WorkerBackend;
use crate::executor::columns::CALL_FIELD_NAMES;
use crate::executor::computations::{Computation, Computations, RankResources, Resources};
use crate::executor::experts::ExpertMember;
use crate::executor::image_inputs::Images;
use crate::executor::latents::LatentLedger;
use crate::executor::pages::{Pages, RelayStaging, warmup_pages};
use crate::executor::python::PythonExecutor;
use crate::executor::reports::Reports;
use crate::ffi::Argument;
use crate::ffi::{Handle, Interpreter};
use crate::job::{BLOCK_ON_TIMEOUT, Job, JobQueue};
use crate::memory::allocation::StorageView;
use crate::memory::{CachePlan, LatentPlan, OutputPool, PrefixCache};
use crate::parallel::world::CollectiveGroup;
use crate::stream::Scratches;
use crate::stream::{CUDAStream, Streams};
use crate::worker::image::{ImageInfo, ImageManifest};
use crate::{Error, LatentPool, LatentStorage, Result};

/// Graph storage keys of the computations' streams: this plus the stream's
/// index among them.
const COMPUTATION_GRAPHS: usize = 1100;

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

/// The executor's `video` section for a video model whose placed denoiser
/// is `denoiser`.
pub(crate) fn video_settings(config: &WorkerConfig, denoiser: &str) -> serde_json::Value {
    serde_json::json!({
        "denoiser": denoiser,
        "resolutions": config.video_resolutions,
        "aspect_ratios": config.video_aspect_ratios,
        "max_seconds": config.max_video_seconds,
        "min_seconds": config.min_video_seconds,
        "text_capacities": config.video_text_capacities,
        "max_condition_rows": config.max_condition_rows,
        "ffmpeg": config.ffmpeg,
        // A conditioned request whose layout startup did not prepare
        // evaluates in a transient one; the denoiser keeps one per slot.
        "transient_layouts": config.max_request_pool_size,
    })
}

/// Whether the deployment's expert union exchanges tokens (a union that
/// prefetches weights instead runs no expert steps).
fn exchanges_tokens(launch: &WorkerLaunch) -> bool {
    launch
        .expert_parallel
        .as_ref()
        .is_some_and(|union| !union.prefetches_weights())
}

/// The executor's `pages` section for a rank holding a causal or
/// diffusion language model component and a prefix cache planned as
/// `cache`; `None` for any other rank.
pub(crate) fn page_settings(
    launch: &WorkerLaunch,
    config: &WorkerConfig,
    components: &Components,
    cache: Option<&CachePlan>,
    image_processor: bool,
    media_axes: Option<usize>,
) -> Result<Option<serde_json::Value>> {
    let holds = components.placed().iter().any(|placed| {
        !placed.host() && (placed.has(Capability::CausalLm) || placed.has(Capability::DiffusionLm))
    });
    let Some(cache) = cache.filter(|_| holds && config.role != "experts") else {
        return Ok(None);
    };
    let groups: Vec<serde_json::Value> = cache
        .group_shapes()?
        .iter()
        .map(|shape| {
            serde_json::json!({
                "block_size": shape.block_size,
                "units_per_page": shape.units_per_page,
                "window": shape.window,
            })
        })
        .collect();
    let union = launch.expert_parallel.as_ref();
    Ok(Some(serde_json::json!({
        "max_batch_calls": config.max_batch_calls,
        "max_batch_tokens": config.max_batch_tokens,
        "max_model_len": config.max_model_len,
        "graph_policy": config.graph_policy,
        "decode_graph_batch_sizes": config.decode_graph_batch_sizes,
        "prefill_graph_token_sizes": config.prefill_graph_token_sizes,
        "prefill_outputs": config.prefill_outputs,
        "draft_tokens": config.draft_tokens,
        "cache_groups": groups,
        "canvas_sampling": config.canvas_sampling,
        "image_processor": image_processor,
        "media_axes": media_axes,
        "expert": {
            "exchanges_tokens": exchanges_tokens(launch),
            "attention_ranks": union.map_or(0, |union| union.attention_ranks),
        },
    })))
}

/// The executor's `experts` section: this rank's membership of an expert
/// union that exchanges tokens; `None` without one.
pub(crate) fn expert_settings(launch: &WorkerLaunch) -> Option<serde_json::Value> {
    let union = launch
        .expert_parallel
        .as_ref()
        .filter(|_| exchanges_tokens(launch))?;
    let execution = &launch.execution;
    let dedicated = execution.role == "experts";
    // A source rank's forwards run together with its worker's other ranks;
    // an expert-only rank runs none.
    let sources: Vec<usize> = if dedicated {
        Vec::new()
    } else {
        let first = union.rank - execution.rank;
        (first..first + execution.world_size).collect()
    };
    Some(serde_json::json!({
        "rank": union.rank,
        "size": union.size,
        "attention_ranks": union.attention_ranks,
        "sources": sources,
        "transport": union.exchange,
        "microbatches": execution.expert_microbatches.max(1),
        "dedicated": dedicated,
    }))
}

/// The components whose computations run collectives on streams of their own,
/// with how many each runs on: a causal language model one per expert
/// microbatch, a diffusion language model one. The same on every rank, which
/// plans the stream groups of every component, held or not.
pub(crate) fn stream_copies(
    launch: &WorkerLaunch,
    declarations: &Declarations,
) -> BTreeMap<String, usize> {
    let microbatches = if exchanges_tokens(launch) {
        launch.execution.expert_microbatches.max(1)
    } else {
        1
    };
    launch
        .components
        .keys()
        .filter_map(|name| {
            let calls = declarations.calls(name);
            let copies = if calls.iter().any(|call| call.is(Capability::CausalLm)) {
                microbatches
            } else if calls.iter().any(|call| call.is(Capability::DiffusionLm)) {
                1
            } else {
                return None;
            };
            Some((name.clone(), copies))
        })
        .collect()
}

/// The sections of a rank's executor settings.
#[derive(Default)]
pub(crate) struct Sections {
    pub(crate) video: Option<serde_json::Value>,
    pub(crate) image: Option<serde_json::Value>,
    pub(crate) pages: Option<serde_json::Value>,
    pub(crate) experts: Option<serde_json::Value>,
}

impl Sections {
    fn is_empty(&self) -> bool {
        self.video.is_none()
            && self.image.is_none()
            && self.pages.is_none()
            && self.experts.is_none()
    }
}

/// One computation as the manifest states it.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ComputationManifest {
    pub(crate) component: String,
    /// The call kinds the computation serves, by their protocol names.
    pub(crate) kinds: Vec<String>,
    /// The executor prepares each batch (`prepare`).
    pub(crate) prepares: bool,
    /// The executor advances the computation's slot state past each batch's
    /// commit point (`commit`).
    pub(crate) commits: bool,
    /// The worker's resources the computation uses: `pages`, `latents`,
    /// `reports`, `images`.
    #[serde(default)]
    pub(crate) resources: Vec<String>,
    /// The expert microbatches the computation's forwards split into.
    #[serde(default = "one")]
    pub(crate) microbatches: usize,
    /// Output words of the computation's largest batch; a call's completion
    /// word each when absent.
    #[serde(default)]
    pub(crate) output_words: usize,
}

fn one() -> usize {
    1
}

/// What the manifest states of the computations over the prefix cache.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct PageManifest {
    /// The call columns' field names in row order, which must be the
    /// worker's.
    pub(crate) call_fields: Vec<String>,
    /// Device bytes the computations allocate beside the cache pool whatever
    /// the slot count, at the batch bounds.
    pub(crate) fixed_bytes: u64,
    /// Device bytes they allocate per request slot.
    pub(crate) slot_bytes: u64,
}

impl PageManifest {
    /// Device bytes the computations allocate beside the cache pool for
    /// `request_slots` slots.
    pub(crate) fn reserved_bytes(&self, request_slots: usize) -> u64 {
        self.fixed_bytes
            .saturating_add(self.slot_bytes.saturating_mul(request_slots as u64))
    }
}

/// What the executor states of the rank (`Executor.manifest()`).
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct Manifest {
    #[serde(default)]
    pub(crate) computations: Vec<ComputationManifest>,
    pub(crate) pages: Option<PageManifest>,
    pub(crate) video: Option<VideoManifest>,
    #[serde(default)]
    pub(crate) image: Option<ImageManifest>,
}

/// Construct the rank's executor over the loaded model with `sections`,
/// and read its manifest; `None` for a rank that serves no section.
pub(crate) fn construct(
    loaded: &Handle,
    config: &WorkerConfig,
    components: &Components,
    attention: &str,
    sections: Sections,
) -> Result<Option<(Arc<PythonExecutor>, Manifest)>> {
    if sections.is_empty() {
        return Ok(None);
    }
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
        "video": sections.video,
        "image": sections.image,
        "pages": sections.pages,
        "experts": sections.experts,
    });
    let executor = PythonExecutor::new(loaded, &settings.to_string())?;
    let manifest: Manifest = serde_json::from_str(&executor.manifest()?)
        .map_err(|error| Error::Interface(format!("the executor's manifest: {error}")))?;
    if let Some(pages) = &manifest.pages
        && pages.call_fields != CALL_FIELD_NAMES
    {
        return Err(Error::Interface(format!(
            "the executor's call fields {:?} are not the worker's {CALL_FIELD_NAMES:?}",
            pages.call_fields
        )));
    }
    Ok(Some((Arc::new(executor), manifest)))
}

/// The call kind a manifest names.
fn kind(name: &str) -> Result<CallKind> {
    CallKind::ALL
        .into_iter()
        .find(|kind| kind.as_str() == name)
        .ok_or_else(|| Error::Interface(format!("the executor names the unknown call {name:?}")))
}

/// The resources a manifest names.
fn resources(names: &[String]) -> Result<Resources> {
    let mut resources = Resources::default();
    for name in names {
        match name.as_str() {
            "pages" => resources.pages = true,
            "latents" => resources.latents = true,
            "reports" => resources.reports = true,
            "images" => resources.images = true,
            other => {
                return Err(Error::Interface(format!(
                    "the executor names the unknown computation resource {other:?}"
                )));
            }
        }
    }
    Ok(resources)
}

/// What [`bind`] binds the executor with.
pub(crate) struct Binding<'a> {
    pub(crate) executor: &'a Arc<PythonExecutor>,
    /// The loaded model, whose image policy prepares inline images.
    pub(crate) loaded: &'a Handle,
    pub(crate) manifest: &'a Manifest,
    pub(crate) experts: Option<&'a ExpertMember>,
    pub(crate) config: &'a WorkerConfig,
    pub(crate) components: &'a Components,
    pub(crate) request_slots: usize,
    pub(crate) queue_depth: usize,
    pub(crate) streams: &'a mut Streams,
    pub(crate) prefix_cache: Option<&'a PrefixCache>,
    /// A video model's capacity, which sizes a latent ledger.
    pub(crate) video: Option<&'a VideoCapacity>,
    /// An image model's latent pool, which sizes its ledger.
    pub(crate) latent: Option<LatentPlan>,
    /// Host tasks the rank admits at once, which bound the image host pool's.
    pub(crate) host_capacity: usize,
    pub(crate) scratches: &'a mut Scratches,
    pub(crate) graphs: &'a mut GraphStorage,
    /// The stream groups this rank belongs to.
    pub(crate) collectives: &'a [CollectiveGroup],
    /// The distributed expert weights' prefetch object, with DWDP.
    pub(crate) prefetch: Option<&'a Handle>,
    /// Where a latent ledger's banks are taken.
    pub(crate) memory: &'a mut dyn crate::memory::ExclusiveMemory,
}

/// What binding leaves: the computations, the most rows a token-denoising step
/// holds, which the engine warms the serving path with, and what an image
/// model's computations report.
pub(crate) struct Bound {
    pub(crate) computations: Option<Computations>,
    pub(crate) step_rows: u32,
    pub(crate) image: Option<ImageInfo>,
}

/// What `bind` returns.
#[derive(Deserialize)]
struct Bind {
    #[serde(default)]
    diffusion_lm_step_rows: u32,
    #[serde(default)]
    image: Option<ImageInfo>,
}

/// The streams the computations compute on, in the order the executor binds
/// them, and which of them the expert microbatches run on.
struct BoundStreams {
    streams: Vec<Stream>,
    /// The microbatches' streams, by index: the device's primary stream, then
    /// its forks.
    microbatches: Vec<usize>,
}

impl BoundStreams {
    /// The index of `stream`, added when new.
    fn index(&mut self, stream: &Stream) -> usize {
        match self
            .streams
            .iter()
            .position(|known| known.raw() == stream.raw())
        {
            Some(index) => index,
            None => {
                self.streams.push(stream.clone());
                self.streams.len() - 1
            }
        }
    }
}

/// Allocate what the plan grants the computations and bind the executor to it:
/// one stream per device computation (the device's primary stream for the
/// computations over the prefix cache, with the expert microbatches' forks, a
/// fork of its own for any other), each with its scratch and, under graphs, a
/// graph pool; the prefix cache's unit pool and block tables with each such
/// computation's relay staging; a latent ledger's banks; the pipe host tasks
/// report through; the image host pool; the stream groups and the weight
/// prefetch object. A rank whose executor has no computation but an expert
/// union binds the union's streams. Under graphs a rank whose modules also run
/// on other devices (a generation tower) binds one more pool per stream on each
/// of them (`peers`), stream by stream after the streams' own.
///
/// The values `bind` receives follow its JSON in this order: each stream's
/// scratch, the host scratch when a computation computes on the host, the graph
/// pools, the prefix cache's unit pool and tables, the relay staging, the
/// latent banks, the stream groups' handles and the prefetch object.
pub(crate) fn bind(start: Binding<'_>) -> Result<Bound> {
    let Binding {
        executor,
        loaded,
        manifest,
        experts,
        config,
        components,
        request_slots,
        queue_depth,
        streams,
        prefix_cache,
        video,
        latent,
        host_capacity,
        scratches,
        graphs,
        collectives,
        prefetch,
        memory,
    } = start;
    if manifest.computations.is_empty() && experts.is_none() {
        return Ok(Bound {
            computations: None,
            step_rows: 0,
            image: None,
        });
    }
    let device = config.device.cuda();
    let max_calls = config.max_batch_calls.max(1);
    let mut known = BoundStreams {
        streams: Vec::new(),
        microbatches: Vec::new(),
    };
    // The expert microbatches' streams come first: the device's primary stream,
    // then its forks.
    for stream in experts.map(ExpertMember::streams).unwrap_or_default() {
        let index = known.index(stream);
        known.microbatches.push(index);
    }

    let mut computations = Vec::with_capacity(manifest.computations.len());
    let mut computations_json = Vec::with_capacity(manifest.computations.len());
    for stated in &manifest.computations {
        let (computation, json) = bind_computation(
            stated,
            components,
            streams,
            &mut known,
            device,
            max_calls,
            queue_depth,
        )?;
        computations.push(computation);
        computations_json.push(json);
    }

    let mut values: Vec<Argument> = Vec::new();
    for stream in &known.streams {
        values.push(Argument::Handle(scratches.get(stream.raw(), device)?));
    }
    // Computations on the host have no stream; they share the rank's host
    // scratch.
    let host_scratch = computations
        .iter()
        .any(|computation| computation.device.is_none());
    if host_scratch {
        values.push(Argument::Handle(scratches.get(0, None)?));
    }
    let capture = config.graph_policy != "off" && device.is_some();
    // The rank's other devices, whose allocations a capture on its streams
    // may make (modules placed on a generation tower).
    let peers: Vec<DeviceId> = config
        .devices()
        .filter_map(crate::config::Device::cuda)
        .filter(|&other| Some(other) != device)
        .collect();
    if let (true, Some(cuda)) = (capture, device) {
        values.extend(graph_pools(graphs, known.streams.len(), cuda, &peers)?);
    }

    // The prefix cache's unit pool and block tables, which the computations
    // over it read, then their relay staging.
    let (pages_json, pages) = match computations
        .iter()
        .find(|computation| computation.resources.pages)
    {
        Some(computation) => {
            let prefix_cache = prefix_cache.ok_or_else(|| {
                Error::Invalid("computations over the prefix cache require a prefix cache".into())
            })?;
            let (json, pages) = bind_pages(prefix_cache, computation, config, &mut values)?;
            (json, Some(pages))
        }
        None => (serde_json::Value::Null, None),
    };
    for staging in computations
        .iter()
        .filter_map(|computation| computation.staging.as_ref())
    {
        values.extend(staging.views().into_iter().map(Argument::Tensor));
    }

    let ledger = latent_ledger(&computations, video, latent, request_slots, memory)?;
    if let Some(ledger) = &ledger {
        values.extend(ledger.banks()?.into_iter().map(Argument::Tensor));
    }

    // The image host pool of a rank whose computations read inline images or
    // encode decoded ones.
    let images = computations
        .iter()
        .any(|computation| computation.resources.images)
        .then(|| Images::new(loaded, device, host_capacity))
        .transpose()?;

    // The pipe host tasks report through, and on a CUDA rank the stream
    // device products they wrote are fenced on.
    let reporting = computations
        .iter()
        .any(|computation| computation.resources.reports);
    let reports = reporting.then(Reports::new).transpose()?;
    let host_stream = match device {
        Some(cuda) if reporting => Some(Arc::new(CUDAStream::create(cuda)?)),
        _ => None,
    };

    let collectives_json: Vec<serde_json::Value> = collectives
        .iter()
        .map(|group| {
            values.push(Argument::Handle(group.handle.clone()));
            serde_json::json!({
                "component": group.component,
                "copy": group.copy,
                "names": group.names,
            })
        })
        .collect();
    if let Some(prefetch) = prefetch {
        values.push(Argument::Handle(prefetch.clone()));
    }

    let memory_json = serde_json::json!({
        "slots": request_slots,
        "max_calls": max_calls,
        "streams": known
            .streams
            .iter()
            .map(|stream| serde_json::json!([stream.raw() as i64, device.map(|device| device.0)]))
            .collect::<Vec<_>>(),
        "host_scratch": host_scratch,
        "pools": capture,
        "peers": if capture { peers.iter().map(|device| device.0).collect() } else { Vec::new() },
        "computations": computations_json,
        "experts": known.microbatches,
        "pages": pages_json,
        "latents": ledger.is_some(),
        "reports": reports.as_ref().map(Reports::descriptor).transpose()?,
        "collectives": collectives_json,
        "prefetch": prefetch.is_some(),
    });
    let bound = executor.bind(memory_json.to_string(), values)?;
    let bound: Bind = serde_json::from_str(&bound)
        .map_err(|error| Error::Interface(format!("the executor's bound computations: {error}")))?;
    let computations = (!computations.is_empty()).then(|| {
        Computations::new(
            Arc::clone(executor),
            computations,
            RankResources {
                pages,
                ledger,
                reports,
                images,
                host_stream,
            },
        )
    });
    Ok(Bound {
        computations,
        step_rows: bound.diffusion_lm_step_rows,
        image: bound.image,
    })
}

/// The computation the manifest states as `stated`, and its entry of `bind`'s
/// JSON. A device computation over the prefix cache computes on the device's
/// primary stream, or on the first `microbatches` of the expert microbatches'
/// streams, with relay staging for `max_calls` calls; any other device
/// computation on a fork of its own. Its output storage holds its largest batch
/// for each of `queue_depth` batches in flight, and one more for a batch whose
/// result was reported before its lease retired.
fn bind_computation(
    stated: &ComputationManifest,
    components: &Components,
    streams: &mut Streams,
    known: &mut BoundStreams,
    device: Option<DeviceId>,
    max_calls: usize,
    queue_depth: usize,
) -> Result<(Computation, serde_json::Value)> {
    let placed = components.get(&stated.component).ok_or_else(|| {
        Error::Interface(format!(
            "the executor states a computation of {:?}, which the rank does not hold",
            stated.component
        ))
    })?;
    let kinds = stated
        .kinds
        .iter()
        .map(|name| kind(name))
        .collect::<Result<Vec<_>>>()?;
    if let Some(missing) = kinds.iter().find(|kind| !placed.kinds.contains(kind)) {
        return Err(Error::Interface(format!(
            "component {:?} serves no {} calls",
            stated.component,
            missing.as_str()
        )));
    }
    let resources = resources(&stated.resources)?;
    if !stated.prepares && !resources.pages {
        return Err(Error::Interface(format!(
            "a computation of {:?} that does not prepare computes over the prefix cache",
            stated.component
        )));
    }
    let (computation_device, indices) = match device {
        Some(cuda) if !placed.host() => {
            let indices = if resources.pages {
                let primary = primary_stream(streams, cuda)?;
                let first = known.index(&primary);
                match stated.microbatches {
                    1 => vec![first],
                    count if count <= known.microbatches.len() => {
                        known.microbatches[..count].to_vec()
                    }
                    count => {
                        return Err(Error::Interface(format!(
                            "a computation of {:?} runs {count} microbatches on {} streams",
                            stated.component,
                            known.microbatches.len()
                        )));
                    }
                }
            } else {
                let fork = streams.fork_module(cuda)?.stream()?.clone();
                vec![known.index(&fork)]
            };
            (Some(cuda), indices)
        }
        _ => (None, Vec::new()),
    };
    let staging = resources
        .pages
        .then(|| RelayStaging::allocate(computation_device, max_calls))
        .transpose()?;
    let json = serde_json::json!({
        "component": stated.component,
        "kinds": stated.kinds,
        "streams": indices,
        "resources": stated.resources,
    });
    let computation = Computation {
        component: stated.component.clone(),
        kinds,
        prepares: stated.prepares,
        commits: stated.commits,
        resources,
        device: computation_device,
        stream: indices.first().map(|&index| known.streams[index].clone()),
        outputs: OutputPool::new(
            computation_device,
            queue_depth.max(1) + 1,
            stated.output_words.max(max_calls),
        )?,
        staging,
    };
    Ok((computation, json))
}

/// One graph pool per computation stream, of `streams`, on `device`, then,
/// stream by stream, one on each of `peers`, each registered with `graphs`
/// under the stream's key.
fn graph_pools(
    graphs: &mut GraphStorage,
    streams: usize,
    device: DeviceId,
    peers: &[DeviceId],
) -> Result<Vec<Argument>> {
    let create = Interpreter::import("uniserve.cuda_graph.pool")?;
    let devices: Vec<DeviceId> = std::iter::once(device)
        .chain(peers.iter().copied())
        .collect();
    let mut own = Vec::with_capacity(streams);
    let mut peer_pools = Vec::new();
    for index in 0..streams {
        let mut pools =
            reserve_pools(graphs, &create, COMPUTATION_GRAPHS + index, &devices, None)?.into_iter();
        let pool = pools
            .next()
            .ok_or_else(|| Error::Invariant("a graph pool was not reserved".into()))?;
        own.push(Argument::Handle(pool.value.clone()));
        peer_pools.extend(pools.map(|pool| Argument::Handle(pool.value.clone())));
    }
    own.extend(peer_pools);
    Ok(own)
}

/// Bind the prefix cache to the computations over it, whose first is
/// `computation`: push its unit pool and block tables to `values` and return
/// their entry of `bind`'s JSON and what the computations share.
fn bind_pages(
    prefix_cache: &PrefixCache,
    computation: &Computation,
    config: &WorkerConfig,
    values: &mut Vec<Argument>,
) -> Result<(serde_json::Value, Pages)> {
    values.push(Argument::Handle(prefix_cache.pool()?));
    let mut tables = Vec::new();
    for (name, view) in prefix_cache.storage().views() {
        tables.push(name);
        values.push(Argument::Tensor(view));
    }
    let json = serde_json::json!({
        "units": prefix_cache.planes().num_units(),
        "tables": tables,
    });
    let pages = Pages::new(
        computation.device,
        computation.stream.clone(),
        prefix_cache.bind_units(computation.stream.as_ref())?,
        config.canvas_sampling,
    );
    Ok((json, pages))
}

/// The latent ledger of the computation that steps latent trajectories, its
/// banks allocated on that computation's stream: a video model's slot pages
/// (`video`), or an image model's pool whose pages the scheduler assigns
/// (`latent`). `None` without such a computation.
fn latent_ledger(
    computations: &[Computation],
    video: Option<&VideoCapacity>,
    latent: Option<LatentPlan>,
    request_slots: usize,
    memory: &mut dyn crate::memory::ExclusiveMemory,
) -> Result<Option<LatentLedger>> {
    let Some(computation) = computations
        .iter()
        .find(|computation| computation.resources.latents)
    else {
        return Ok(None);
    };
    // A video slot owns a fixed run of pages; an image trajectory the pages
    // its calls name.
    let (plan, slot_pages) = match (video, latent) {
        (Some(capacity), _) => (
            capacity.latent(request_slots),
            Some(capacity.manifest.pages),
        ),
        (None, Some(plan)) => (plan, None),
        (None, None) => {
            return Err(Error::Unsupported(
                "a latent ledger requires a video model's capacity or an image model's \
                 latent pool"
                    .into(),
            ));
        }
    };
    // The banks live on the computation's device, or in host memory on a rank
    // without one.
    let storage = LatentStorage::allocate(
        &plan,
        computation.device,
        computation.stream.as_ref(),
        "computations.latent_banks",
        memory,
    )?;
    let pool = LatentPool::new(plan.request_pool_size, plan.num_pages, plan.page_units)?;
    Ok(Some(match slot_pages {
        Some(pages) => LatentLedger::new(pool, storage, pages.page_units, pages.pages),
        None => LatentLedger::scheduled(pool, storage),
    }))
}

/// The device's primary stream.
fn primary_stream(streams: &Streams, device: DeviceId) -> Result<Stream> {
    Ok(streams
        .stream_for(device)
        .ok_or(Error::State("the device has no primary stream"))?
        .stream()?
        .clone())
}

/// One warmup phase as the executor lays it out (`warmup_phases`).
#[derive(Debug, Deserialize)]
struct Phase {
    computation: usize,
    /// The scratch tables' lengths, of slots `1..`; none for a phase that
    /// reads no prefix cache.
    lengths: Option<Vec<usize>>,
    /// Each step's attention rows, `[slot, prefix, query]`.
    #[serde(default)]
    steps: Vec<Vec<[u32; 3]>>,
    /// Whether the steps' rows write the cache; by default those of a
    /// computation whose batches write it do.
    #[serde(default)]
    writes: Option<bool>,
}

/// The warmup of the executor's computations and expert union: the phases the
/// executor lays out, each a job, on scratch tables the worker installs and
/// releases around it when it reads the prefix cache; then its end (the
/// captures that follow every phase and the end of the union's announced
/// steps). Run again by a promoted standby whose expert workspaces were
/// rebuilt. The serving path itself is warmed by the engine once the rank
/// reported, with real requests of the shapes the rank states
/// (`WorkerInfo::image_warmups`, `diffusion_lm_step_rows`).
pub(crate) fn warm(backend: &mut WorkerBackend) -> std::result::Result<(), WorkerError> {
    run_phases(backend).map_err(|error| {
        WorkerError::classify(error, Some("computation warmup")).with_phase("warmup")
    })
}

/// Run every warmup phase, then the warmup's end.
fn run_phases(backend: &mut WorkerBackend) -> Result<()> {
    let executor = Arc::clone(
        backend
            .executor()
            .ok_or(Error::State("computation warmup has no executor"))?,
    );
    let jobs = backend.job_queue().clone();
    let phases: Vec<Phase> = serde_json::from_str(&executor.warmup_phases()?)
        .map_err(|error| Error::Interface(format!("the executor's warmup phases: {error}")))?;
    let stream = backend.computations().and_then(Computations::pages_stream);
    for (index, phase) in phases.into_iter().enumerate() {
        let writes = phase.writes.unwrap_or_else(|| {
            backend
                .computations()
                .is_some_and(|computations| computations.writes_pages(phase.computation))
        });
        let Some(lengths) = phase.lengths else {
            if !phase.steps.is_empty() {
                return Err(Error::Interface(format!(
                    "warmup phase {index} attends without scratch tables"
                )));
            }
            perform(&jobs, phase_job(&executor, index, Vec::new()))?;
            continue;
        };
        let prefix_cache = backend
            .worker_storage_mut()
            .and_then(|storage| storage.prefix_cache.as_mut())
            .ok_or(Error::State(
                "a warmup phase over scratch tables requires a prefix cache",
            ))?;

        // One job installs the scratch tables, one runs the phase with each
        // step's page columns, and one clears them and resets the scratch
        // units, whether or not the phase succeeded.
        let before = Job::new();
        let scratch = prefix_cache.install_scratch(&lengths, before.on(stream.as_ref()))?;
        perform(&jobs, before)?;
        let ran = warmup_pages(prefix_cache, &phase.steps, writes)
            .and_then(|pages| perform(&jobs, phase_job(&executor, index, pages)));
        let after = Job::new();
        let released =
            prefix_cache.release_scratch(scratch, lengths.len(), after.on(stream.as_ref()));
        let performed = perform(&jobs, after);
        ran.and(released).and(performed)?;
    }
    let finish = Job::new();
    finish.call(executor.warmup_finish());
    perform(&jobs, finish)
}

/// The job that runs warmup phase `index` with its steps' page columns.
fn phase_job(executor: &PythonExecutor, index: usize, pages: Vec<StorageView>) -> Job {
    let job = Job::new();
    job.call(executor.warmup_phase(index, pages));
    job
}

/// Perform `job` on the builder thread and wait for it.
fn perform(jobs: &JobQueue, job: Job) -> Result<()> {
    jobs.block_on(job, BLOCK_ON_TIMEOUT)?
        .check()
        .map(drop)
        .map_err(|failure| failure.error)
}

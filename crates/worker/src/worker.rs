//! The root owner of a rank: construction, warmup, serving and close.
//!
//! [`Worker::start`] builds a rank in a fixed order, so every collective
//! happens in the same order on every rank and a failure releases exactly
//! what was built:
//!
//! 1. Device check and primary context.
//! 2. Model description and placement validation (no collective yet), then
//!    checkpoint narrowing.
//! 3. The process world: host groups, the torch world, component groups
//!    and meshes.
//! 4. Loading, model facts, the block size and cache layout, the rank's
//!    executor and its manifest, execution streams and the expert union.
//! 5. Capacity: request slots, cache units and latent pages, all agreed over
//!    the group; the auxiliary-device check.
//! 6. Request storage, the stream groups and the executor's computations bound
//!    on the plan, handed to the backend; an image model's ranks agree on their
//!    graph keys.
//! 7. The batch executor and the profiler.
//!
//! [`Worker::warmup`] runs the warmup before the rank reports; the engine then
//! warms the serving path with real requests of the shapes the report states.
//! [`Worker::run`] serves the channel, and [`Worker::close`] consumes the rank
//! and releases its owners in dependency order: execution, the executor and its
//! computations, streams, requests, the process world, then deferred device
//! memory. The launch closes every rank that started, unless the rank abandoned
//! work its threads may wait in for good, which ends the process at once.

mod computations;
mod image;
mod loading;
mod memory;
mod planning;
pub(crate) mod update;

pub(crate) use loading::LoadedModel;
pub(crate) use memory::WorkerMemory;

use std::sync::Arc;

use std::collections::BTreeMap;

use tvm_ffi::Any;

use cuda::{DeviceId, PrimaryContext};
use ipc::ComponentWeights;
use ipc::{CallKind, Channel, WorkerEndpoint, WorkerInfo};

use crate::capacity::checks;
use crate::capacity::device::{DeviceBudget, total_bytes};
use crate::capacity::plan::Plan;
use crate::config::placement::{Capability, Components, GroupPlacement};
use crate::config::{Device, ExpertUnion, WorkerConfig, WorkerLaunch};
use crate::cuda_graph::{GraphStorage, graph_storage_budget_bytes};
use crate::error::WorkerError;
use crate::executor::Executor;
use crate::executor::backend::WorkerBackend;
use crate::executor::experts::{ExpertMember, ExpertStart};
use crate::executor::transfer::TransferRuntime;
use crate::ffi::{Handle, Interpreter};
use crate::job::{BLOCK_ON_TIMEOUT, JobQueue};
use crate::memory::allocation::Storage;
use crate::memory::prefix_cache::host::{self, HostCacheSetup};
use crate::memory::retained::Retained;
use crate::memory::weights::{self, WeightKeys};
use crate::parallel::world::{World, startup_timeout};
use crate::parallel::{DwdpManager, HostRetention, HostWeights};
use crate::profiling::kernels::{KernelRecords, KernelWatch};
use crate::profiling::profiler::{ProfileConfig, Profiler};
use crate::profiling::range;
use crate::service::{Ending, Parked, ParkedExperts, Service, Suspension};
use crate::stream::Scratches;
use crate::stream::Streams;
use crate::transfer::TransportKind;
use crate::{Error, Result};
use loading::{ComponentMesh, Description, ExpertPartition, head_projection, sequence_kind};

use computations::Binding;
pub(crate) use computations::warm as warm_computations;
use memory::allocate_storage;
use planning::{block_size, cache_plan, cache_width, log_capacity, plan};

/// One caching-allocator segment: device, graph pool id and bytes.
type Segment = (DeviceId, (u64, u64), u64);

/// A startup failure, with the phase that failed.
fn phase(phase: &'static str) -> impl Fn(Error) -> WorkerError {
    move |error| {
        // Description and process-group setup refuse a launch the rank
        // cannot serve; report input errors there as unsupported setups.
        // `ValueError` is the kind of a returned value TVM-FFI could not
        // convert.
        let error = match error {
            Error::Numerical {
                kind,
                message,
                traceback,
                function,
            } if matches!(phase, "describe" | "process groups")
                && matches!(kind.as_str(), "InputError" | "FileNotFound" | "ValueError") =>
            {
                Error::Numerical {
                    kind: "UnsupportedSetup".into(),
                    message,
                    traceback,
                    function,
                }
            }
            error => error,
        };
        WorkerError::classify(error, Some(phase)).with_phase(phase)
    }
}

/// A rank.
pub(crate) struct Worker {
    config: WorkerConfig,
    info: WorkerInfo,
    plan: Plan,
    model: LoadedModel,
    world: World,
    budget: Option<DeviceBudget>,
    streams: Streams,
    graphs: GraphStorage,
    /// The shared work areas of the computations' streams, which the executor's
    /// execution contexts borrow.
    scratches: Scratches,
    /// How the worker, parked as a standby, keeps its graph memory.
    standby_memory: ipc::StandbyMemory,
    executor: Executor,
    /// The executor runs warmup phases: the rank has executor computations or
    /// an expert union member.
    computation_warmup: bool,
    /// Retained for the rank's lifetime: torch, the driver owners and the
    /// reclaimer all use the device's primary context.
    context: Option<PrimaryContext>,
    /// The worker's connection to its host's Storage Server.
    retained: Arc<Retained>,
    /// The retained storages of the model's weights, mapped for the
    /// worker's lifetime.
    weights: Vec<Arc<Storage>>,
    /// Their keys; `None` for weights the worker does not retain.
    weight_keys: Option<WeightKeys>,
    /// The identity of what the loaded model's forwards compute
    /// ([`weights::model_identity`]), which keys its cache pages.
    identity: String,
    /// The launch the worker loaded its model with.
    launch: WorkerLaunch,
    /// What holds the plugins' communication buffers, which serve for the
    /// worker's lifetime.
    _plugin_buffers: Option<Handle>,
    /// What mapping the rank's host page pools needs, until the engine
    /// sizes them; `None` for a rank that copies no page through the host
    /// KV cache.
    host_cache: Option<HostCacheSetup>,
    warmed: bool,
}

impl Worker {
    /// Build the rank. On failure, everything built so far is released;
    /// the process world is abandoned, because peers may still be inside a
    /// collective.
    pub(crate) fn start(mut launch: WorkerLaunch) -> std::result::Result<Self, WorkerError> {
        let device = launch.device();
        let context = match device {
            Device::Cuda(cuda) => {
                let count = DeviceId::count().map_err(|error| phase("device")(error.into()))?;
                if cuda.0 >= count {
                    return Err(phase("device")(Error::Unsupported(format!(
                        "cuda device {device} is outside the {count} visible CUDA device(s)"
                    ))));
                }
                Some(PrimaryContext::retain(cuda).map_err(|error| phase("device")(error.into()))?)
            }
            Device::Cpu => None,
        };
        // The worker's memory that outlives it is kept at the host's
        // Storage Server, which first shoots down a predecessor still
        // holding it.
        let retained = Retained::connect(
            &launch.storage,
            &launch.group,
            launch.execution.rank as u32,
            launch.generation,
        )
        .map_err(phase("storage"))?;

        let describing = range(c"uniserve.worker.describe", None);
        let description = Description::describe(&launch).map_err(phase("describe"))?;

        drop(describing);
        let mut world =
            World::initialize(&launch, startup_timeout()).map_err(phase("process groups"))?;
        let dedicated = launch.execution.role == "experts";
        if !dedicated {
            world
                .bind_components(&launch, &description.declarations)
                .map_err(phase("process groups"))?;
        }

        let components = Components::new(
            &description.declarations,
            &launch.components,
            launch.execution.rank,
            device,
            |name| world.meshes(name),
        );
        let host_only = !dedicated && components.host_only();
        let meshes: Vec<ComponentMesh> = components
            .placed()
            .iter()
            .filter_map(|placed| {
                placed.mesh.clone().map(|mesh| ComponentMesh {
                    component: placed.name.clone(),
                    paths: description.declarations.roots(&placed.name),
                    mesh,
                    units: placed.units.clone(),
                    sequence: sequence_kind(&placed.config),
                    projection: head_projection(&placed.config),
                })
            })
            .collect();
        let experts = launch
            .expert_parallel
            .as_ref()
            .map(|union| ExpertPartition {
                world: world.torch().clone(),
                attention_ranks: union.attention_ranks,
                dedicated,
            });
        let loading = range(c"uniserve.worker.load", None);
        // The weights the rank retained when it last loaded this model on
        // this placement bind instead of loading. Loading is collective, so
        // the ranks that load together read what any of them misses
        // (`weights::Load`), and bind the rest.
        let fingerprints =
            model_fingerprints(&launch, &description, host_only).map_err(phase("load"))?;
        let keys = weight_keys(&launch, fingerprints.as_ref(), &retained).map_err(phase("load"))?;
        let found = match &keys {
            Some(keys) => keys.find(&retained).map_err(phase("load"))?,
            None => None,
        };
        let agreement = world.union.as_ref().unwrap_or(&world.host);
        let load =
            agree_load(agreement, weights::Load::of(found.as_ref())).map_err(phase("load"))?;
        let (found, load) = match load {
            weights::Load::Tensors(names) => (found, names),
            weights::Load::All => (None, Default::default()),
        };
        // Weights the worker reads take the place of the scope's others.
        if keys.is_some() && (found.is_none() || !load.is_empty()) {
            retained.release_read_only().map_err(phase("load"))?;
        }
        let binding = found
            .as_ref()
            .map(|found| found.bind(&load))
            .transpose()
            .map_err(phase("load"))?;
        let model = LoadedModel::load(
            description,
            &launch,
            meshes,
            experts,
            host_only,
            binding.as_ref(),
            &load,
        )
        .map_err(phase("load"))?;
        drop(binding);
        let identity = weights::model_identity(model.handle()).map_err(phase("load"))?;
        // A loading worker's weights are put once the host regions of the
        // layers it streams from host memory joined them.
        let host_paths: Vec<String> = components
            .placed()
            .iter()
            .filter(|placed| placed.config.weights == ComponentWeights::Host)
            .flat_map(|placed| model.declarations.roots(&placed.name))
            .collect();
        // A rank publishes what it read: everything, or the tensors it
        // read beside those it bound.
        let (mut published, bound) = match (found, &keys) {
            (Some(found), Some(keys)) if !load.is_empty() => (
                Some(
                    keys.publish(model.handle(), &host_paths, Some(&load))
                        .map_err(phase("load"))?,
                ),
                found.into_storages(),
            ),
            (Some(found), _) => (None, found.into_storages()),
            (None, Some(keys)) => (
                Some(
                    keys.publish(model.handle(), &host_paths, None)
                        .map_err(phase("load"))?,
                ),
                Vec::new(),
            ),
            (None, None) => (None, Vec::new()),
        };
        // Every later graph decision of the rank reads the resolved policy.
        let host_component = components
            .placed()
            .iter()
            .find(|placed| placed.config.weights == ComponentWeights::Host)
            .map(|placed| placed.name.as_str());
        plugin_graph_policy(&mut launch.execution).map_err(phase("load"))?;
        launch.execution.graph_policy =
            graph_policy(&launch.execution.graph_policy, &model, host_component)
                .map_err(phase("load"))?;
        let mut config = launch.execution.clone();
        if model.video_decoder {
            config = config
                .for_media_slots(launch.ipc.queue_depth)
                .map_err(phase("load"))?;
        }
        // How the computations over the prefix cache lay out a model's image
        // features: a model without image capabilities places none.
        let image_capable = [
            Capability::ImageDenoiser,
            Capability::PatchEncoder,
            Capability::PatchAutoencoder,
            Capability::ImageDecoder,
        ]
        .into_iter()
        .any(|capability| model.declarations.has(capability));
        let text_facts = if image_capable {
            image::TextFacts::of(model.handle()).map_err(phase("load"))?
        } else {
            image::TextFacts::default()
        };
        // The executor's token computations are sized by the cache's layout, so
        // the block size is resolved first, collectively over the worker group.
        block_size(&mut config, &model, &world).map_err(phase("capacity"))?;
        let cache = cache_plan(&launch, &config, &model).map_err(phase("capacity"))?;
        // The rank's executor states what capacity planning sizes the rank by:
        // a video model's slot layout, the storage of its computations over the
        // prefix cache.
        let video_denoiser = if model.video_decoder {
            Some(
                computations::placed_denoiser(&model.declarations, &config.deployment_components)
                    .map_err(phase("load"))?,
            )
        } else {
            None
        };
        let sections = computations::Sections {
            video: video_denoiser
                .as_deref()
                .map(|denoiser| computations::video_settings(&config, denoiser)),
            image: image::holds(&components, video_denoiser.is_some())
                .then(|| image::settings(&config)),
            pages: computations::page_settings(
                &launch,
                &config,
                &components,
                cache.as_ref(),
                text_facts.image_processor,
                text_facts.media_axes,
            )
            .map_err(phase("load"))?,
            experts: computations::expert_settings(&launch),
        };
        let executor = computations::construct(
            model.handle(),
            &config,
            &components,
            &model.attention_backend,
            sections,
        )
        .map_err(phase("load"))?;
        let (executor, manifest) = match executor {
            Some((executor, manifest)) => (Some(executor), manifest),
            None => (None, computations::Manifest::default()),
        };
        let video = match (video_denoiser, &manifest.video) {
            (Some(denoiser), Some(video)) => Some((denoiser, video.clone())),
            (Some(_), None) => {
                return Err(phase("load")(Error::Interface(
                    "the executor's manifest states no video section".into(),
                )));
            }
            (None, _) => None,
        };

        drop(loading);

        let devices: Vec<Device> = config.devices().collect();
        let mut streams = Streams::create(&devices).map_err(phase("streams"))?;
        let host = launch.host.clone();
        // Expert transports map their device workspaces before capacity
        // planning measures free memory.
        let experts = match &executor {
            Some(executor) => ExpertMember::start(ExpertStart {
                launch: &launch,
                world: &world,
                executor,
                streams: &mut streams,
                host: &host,
            })
            .map_err(phase("experts"))?,
            None => None,
        };
        let weights = DwdpManager::start(&launch, &world, &model, &streams, &host)
            .map_err(phase("experts"))?;
        // Host-resident layer weights release their device storage before
        // capacity planning measures free memory.
        // A stack's host region is published with a whole load and bound
        // otherwise: a rank that misses any of it loads whole (`Load::of`).
        let mut retention = match &mut published {
            Some(published) if bound.is_empty() => HostRetention::Publish(published),
            _ if !bound.is_empty() => HostRetention::Bound(&bound),
            _ => HostRetention::None,
        };
        let host_weights =
            HostWeights::start(&model, &components, &mut retention).map_err(phase("load"))?;
        let weight_storages = match published {
            Some(published) => {
                let mut storages = published.put(&retained).map_err(phase("load"))?;
                storages.extend(bound);
                storages
            }
            None => bound,
        };
        // Plugins allocate the memory their kernels communicate through
        // before capacity planning measures free memory.
        let (plugin_buffers, unattributed) =
            allocate_plugin_buffers(device).map_err(phase("capacity"))?;
        let budget = match device {
            Device::Cuda(_) => Some(
                DeviceBudget::new(
                    Interpreter::import("torch.cuda.synchronize").map_err(phase("capacity"))?,
                    Interpreter::import("torch.cuda.empty_cache").map_err(phase("capacity"))?,
                    Arc::clone(&retained),
                    unattributed,
                )
                .map_err(phase("capacity"))?,
            ),
            Device::Cpu => None,
        };

        let planning = range(c"uniserve.worker.plan", None);
        let (plan, video_capacity) = plan(
            &launch,
            &mut config,
            &model,
            manifest.image.as_ref(),
            text_facts.image_processor,
            &world,
            budget.as_ref(),
            &components,
            cache,
            manifest.pages.as_ref(),
            video
                .as_ref()
                .map(|(denoiser, manifest)| (denoiser.as_str(), manifest.clone())),
        )
        .map_err(phase("capacity"))?;
        if let Some(budget) = &budget {
            checks::auxiliary(&config, &plan.fixed_device_bytes, budget)
                .map_err(phase("capacity"))?;
        }
        drop(planning);
        let mut info = plan.info.clone();
        info.validate()
            .map_err(|error| phase("capacity")(Error::Invariant(error.to_string())))?;
        log_capacity(&config, &plan);

        let budgets = devices
            .iter()
            .filter_map(|device| device.cuda())
            .map(|index| {
                Ok((
                    index,
                    graph_storage_budget_bytes(total_bytes(index)? as i64),
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()
            .map_err(phase("capacity"))?;
        let mut graphs = GraphStorage::new(budgets);
        let mut scratches = Scratches::default();

        let storage = allocate_storage(&launch, &config, &plan, &streams, &mut &*retained)
            .map_err(phase("storage"))?;
        // The engine copies the cache's pages through the host KV cache when
        // it says so; pages are keyed by the whole model's fingerprint and
        // the identity of what its forwards compute.
        let model_fingerprint = fingerprints
            .as_ref()
            .map(|fingerprints| fingerprints.pages(&identity));
        // The engine roots the prefix hash chains of the rank's pages at the
        // same identity, so a world computing other numerics, or with other
        // weights, hits none of them.
        if let Some(reported) = info.prefix_cache.as_mut() {
            reported.identity = model_fingerprint
                .clone()
                .unwrap_or_else(|| identity.clone());
        }
        // The engine sizes the host KV cache of every rank of the world at
        // one token count once each reported what a token of its pages
        // takes, and the rank maps its page pools then
        // (`uniserve.host_cache`).
        let host_cache = match (
            launch.storage.host_kv_cache,
            model_fingerprint,
            storage.prefix_cache.as_ref(),
            device.cuda(),
            info.prefix_cache.as_mut(),
        ) {
            (true, Some(fingerprint), Some(prefix_cache), Some(cuda), Some(reported)) => {
                // The host KV cache's copies run on a copy stream of the
                // cache's device, beside the forwards on the primary stream.
                let copy = streams
                    .create_copy(cuda)
                    .and_then(|copy| Ok(copy.stream()?.clone()))
                    .map_err(phase("streams"))?;
                reported.host_token_bytes = host::token_bytes(prefix_cache.plan());
                reported.storage_endpoint = launch.storage.tcp.clone();
                Some(HostCacheSetup {
                    fingerprint,
                    endpoint: launch.storage.tcp.clone(),
                    device: cuda,
                    copy,
                })
            }
            _ => None,
        };
        // Each stream a component's collectives run on gets torch groups of
        // its own, created by every rank at this point of its startup.
        let collectives = if dedicated {
            Vec::new()
        } else {
            world
                .bind_stream_groups(
                    &launch,
                    &model.declarations,
                    &computations::stream_copies(&launch, &model.declarations),
                )
                .map_err(phase("bind"))?
        };
        // The executor's computations and expert union bind, in this order on
        // every rank.
        let bound = match &executor {
            Some(executor) => computations::bind(Binding {
                executor,
                loaded: model.handle(),
                manifest: &manifest,
                experts: experts.as_ref(),
                config: &config,
                components: &components,
                request_slots: info.request_slots as usize,
                queue_depth: launch.ipc.queue_depth,
                streams: &mut streams,
                prefix_cache: storage.prefix_cache.as_ref(),
                video: video_capacity.as_ref(),
                latent: plan.latent,
                host_capacity: info.host_task_capacity as usize,
                scratches: &mut scratches,
                graphs: &mut graphs,
                collectives: &collectives,
                prefetch: weights.as_ref().map(DwdpManager::prefetch),
                memory: &mut &*retained,
            })
            .map_err(phase("bind"))?,
            None => computations::Bound {
                computations: None,
                step_rows: 0,
                image: None,
            },
        };
        // Every rank of an image model captures the same graph keys in the
        // same order, which warmup's collectives require.
        image::agree(
            &bound
                .image
                .as_ref()
                .map(|image| image.digests.clone())
                .unwrap_or_default(),
            &world.host,
        )
        .map_err(phase("bind"))?;
        // The shapes the engine warms the serving path with, through its
        // scheduler, once the rank reported, the image-unconditional
        // guidance prefix it reserves and the image extents it admits.
        let info = {
            let mut info = info;
            if let Some(image) = &bound.image {
                image.apply(&mut info);
            }
            info.diffusion_lm_step_rows = bound.step_rows;
            info
        };
        let placement = GroupPlacement {
            components: launch.components.clone(),
            world_size: config.world_size,
        };
        let mut backend = WorkerBackend::new(info.clone(), config.rank, placement, components)
            .map_err(phase("storage"))?;
        backend.set_worker_storage(storage);
        backend.set_job_queue(JobQueue::default());
        if let Some(executor) = executor {
            backend.set_executor(executor);
        }
        // The executor's computations and the expert union warm through the
        // executor's phases.
        let computation_warmup = bound.computations.is_some() || experts.is_some();
        if let Some(computations) = bound.computations {
            backend.set_computations(computations);
        }
        if let Some(weights) = weights {
            backend.set_weights(weights);
        }
        backend.set_host_weights(host_weights);
        if let Some(experts) = experts {
            backend.set_experts(experts);
        }
        let cuda: Vec<DeviceId> = devices.iter().filter_map(|device| device.cuda()).collect();
        backend.set_transfer(
            TransferRuntime::start(
                &streams,
                &cuda,
                launch.ipc.queue_depth,
                info.max_batch_calls as usize,
            )
            .map_err(phase("bind"))?,
        );
        let profiler = ProfileConfig::from_env(|name| std::env::var(name).ok())
            .map_err(phase("profiling"))?
            .map(|profile| {
                Ok::<_, Error>(Profiler::new(
                    profile,
                    Interpreter::import("uniserve.profiling.start_profiler")?,
                    Interpreter::import("uniserve.profiling.stop_profiler")?,
                    device.cuda(),
                ))
            })
            .transpose()
            .map_err(phase("profiling"))?;
        backend.set_profiler(profiler);
        let collective = world
            .components
            .iter()
            .filter(|meshes| meshes.plan.member)
            .flat_map(|meshes| meshes.plan.mesh.iter().chain(&meshes.plan.units))
            .flat_map(|binding| &binding.groups)
            .any(|group| group.members.len() > 1);
        let executor = Executor::new(
            backend,
            launch.ipc.queue_depth,
            config.world_size > 1,
            collective,
        )
        .map_err(|error| error.with_phase("executor"))?;

        // Every owner took its memory: retained storages of the scope that
        // none took any more are released.
        retained.retain_used().map_err(phase("storage"))?;
        tracing::info!("loaded numerical model");
        Ok(Self {
            config,
            info,
            plan,
            model,
            world,
            budget,
            streams,
            graphs,
            scratches,
            standby_memory: launch.storage.standby_memory,
            executor,
            computation_warmup,
            context,
            retained,
            weights: weight_storages,
            weight_keys: keys,
            identity,
            launch,
            _plugin_buffers: plugin_buffers,
            host_cache,
            warmed: false,
        })
    }

    /// Run warmup once, before the first serving submission, in an order
    /// identical on every rank, since captures run collectives: bind each CUDA
    /// device's graph budget (what its graph pools hold now plus the grant left
    /// after product storage not yet resident), run the executor's warmup
    /// phases ([`warm_computations`]), then check graph residency, report the
    /// kernels and seal the graph storage.
    pub(crate) fn warmup(&mut self) -> std::result::Result<(), WorkerError> {
        if self.warmed {
            return Ok(());
        }
        let executor = &mut self.executor;
        if executor.started() {
            return Err(Error::State("warmup must precede the first serving submission").into());
        }
        let numerical = self.config.role == "experts"
            || executor
                .backend()
                .components()
                .placed()
                .iter()
                .any(|placed| !placed.calls.is_empty());
        let cuda = self.config.device.is_cuda();

        let _warmup = range(c"uniserve.worker.warmup", None);
        if numerical && cuda {
            self.bind_graph_budgets().map_err(phase("warmup"))?;
        }
        let executor = &mut self.executor;
        if self.computation_warmup {
            let _capture = range(c"uniserve.worker.capture", None);
            warm_computations(executor.backend_mut())?;
        }
        let backend = executor.backend();
        if backend.components().holds_host() {
            self.require_codecs()?;
        }
        self.complete_startup()?;

        if let Some(budget) = &self.budget {
            checks::startup(
                &self.config,
                self.plan.arena.deferred_device_bytes(),
                |_| 0,
                budget,
            )
            .map_err(phase("warmup"))?;
        }
        // Serving runs Python only inside library calls; collection pauses
        // there would stall steps, so the heap built so far is collected and
        // frozen and the cyclic collector is disabled: resident models and
        // graphs are permanent, and a collection inside a serving step would
        // traverse them.
        for path in ["gc.collect", "gc.freeze", "gc.disable"] {
            Interpreter::import(path)
                .and_then(|function| function.invoke(&[]))
                .map_err(phase("warmup"))?;
        }
        self.warmed = true;
        Ok(())
    }

    /// Serve the channel until the engine closes the rank or a fatal response
    /// ends admission, or until the engine's channel fails ([`Ending`]),
    /// reporting a job the builder thread stays inside for `stall_bound`.
    pub(crate) fn run(
        &mut self,
        channel: &mut Channel,
        stall_bound: std::time::Duration,
    ) -> std::result::Result<Ending, WorkerError> {
        self.warmup()?;
        // The rank reports the scratch units its warmup used, which a
        // standby worker's warmup stays inside while its predecessor serves.
        let prefix_cache = self
            .executor
            .backend()
            .worker_storage()
            .and_then(|storage| storage.prefix_cache.as_ref());
        let scratch = prefix_cache.map_or(0, crate::memory::PrefixCache::scratch_lent);
        // The planes' segments, the ones they hold (a successor creates its
        // tail as its memory frees) and the units they map from the
        // predecessor world, which the engine allocates only once the world
        // took them.
        let segments = prefix_cache.map(|cache| {
            let planes = cache.planes();
            (
                planes.segments().unwrap_or_default(),
                planes.held_segments(),
                planes.inherited(),
            )
        });
        if let Some(cache) = self.info.prefix_cache.as_mut() {
            cache.scratch_units = scratch;
            if let Some((segments, held, inherited)) = segments {
                cache.segments = segments;
                cache.held_segments = held;
                cache.inherited_start = inherited.start;
                cache.inherited_units = inherited.end - inherited.start;
            }
        }
        // A standby worker parks with its exclusive storages read-only until
        // the engine promotes it, and with the memory its graphs address
        // released when keeping it would crowd the workers it stands by for.
        let standby = if self.retained.standing_by() {
            // A standby that cannot park fails its startup: the engine,
            // still awaiting its startup report, names the error.
            let parked = self.park_memory().and_then(|suspension| {
                self.retained.park().map_err(phase("standby"))?;
                Ok(suspension)
            });
            let suspension = match parked {
                Ok(suspension) => suspension,
                Err(error) => {
                    crate::launch::report_startup_failure(channel, &error);
                    return Err(error);
                }
            };
            Some(Parked {
                retained: Arc::clone(&self.retained),
                suspension,
            })
        } else {
            None
        };
        // The engine routes the calls the rank's role leaves to the groups
        // of the other role; the rank serves them all.
        let mut info = self.info.clone();
        info.supported_calls
            .retain(|&kind| !left_to_other_role(&self.config.role, kind));
        let executor = &mut self.executor;
        // Ranks of an expert union join every step outside their own.
        let experts = executor.backend().experts().is_some();
        executor.backend().set_storage_wake(Some(channel.wake()));
        let mut pools = self.graphs.pools();
        pools.extend(self.scratches.pools());
        let observed = self
            .streams
            .observed()
            .map(|(name, stream)| Ok((name, stream.stream()?.clone())))
            .collect::<Result<Vec<_>>>()?;
        let mut maintenance = crate::service::Maintenance {
            retained: Some(Arc::clone(&self.retained)),
            weights: Some(update::WeightUpdate {
                launch: self.launch.clone(),
                model: self.model.handle().clone(),
                storages: self.weights.clone(),
                keys: self.weight_keys.clone(),
                identity: self.identity.clone(),
            }),
            pools,
            asleep: false,
            cupti: crate::profiling::cupti::Cupti::new(observed),
            predecessor: self.config.predecessor.clone(),
            max_total_tokens: self.config.max_total_tokens,
            devices: self.config.devices().filter_map(Device::cuda).collect(),
            host_cache: self.host_cache.take(),
        };
        // The rank serves with CUPTI attached (`crate::profiling::cupti`),
        // unless a profiler holds it; the control thread's (this one's)
        // waits for the device are counted while it is.
        if let Err(error) = maintenance.cupti.attach() {
            tracing::warn!("CUPTI is not attached: {error}");
        }
        let _control = cuda::cupti::Watched::enter();
        let served = Service::new(
            executor,
            channel,
            info,
            experts,
            stall_bound,
            standby,
            maintenance,
        )
        .run();
        executor.backend().set_storage_wake(None);
        served
    }

    /// How a parked standby keeps the device memory its graphs address, as
    /// `worker_config.standby_memory` says: resident or suspended (its
    /// device memory released, its addresses kept, restored at the
    /// promotion); with `auto`, resident when
    /// every device keeps, beside it, the share of its memory a serving
    /// worker's capacity plan leaves for graphs and activations
    /// ([`graph_storage_budget_bytes`]), and suspended otherwise. The ranks
    /// that park together (the expert union the rank belongs to, else its
    /// worker group) take one mode: one rank's crowded device suspends them
    /// all. Suspending copies the graph pools and scratch to host memory,
    /// has the executor release the expert transports' workspaces
    /// (`park_experts`: those that keep their addresses are suspended, the
    /// others freed and their graphs captured again at the promotion), and
    /// suspends every NCCL communicator of the torch world, the stream
    /// groups' included. Returns what it released.
    ///
    /// # Errors
    ///
    /// Fails when a device lacks that share even once the memory is
    /// suspended: the standby does not fit beside the workers it stands by
    /// for, and none is kept.
    fn park_memory(&mut self) -> std::result::Result<Suspension, WorkerError> {
        let mut pools = self.graphs.pools();
        pools.extend(self.scratches.pools());
        let mut devices: Vec<DeviceId> = pools.iter().map(|pool| pool.device).collect();
        devices.sort_unstable();
        devices.dedup();
        let short = |device: DeviceId| -> Result<bool> {
            let memory = device.memory_info()?;
            Ok(memory.free < graph_storage_budget_bytes(memory.total as i64))
        };
        let bytes: usize = pools.iter().map(|pool| pool.memory.mapped_bytes()).sum();
        let crowded = match self.standby_memory {
            ipc::StandbyMemory::Resident => false,
            ipc::StandbyMemory::Suspended => true,
            ipc::StandbyMemory::Auto => {
                let mut crowded = false;
                for &device in &devices {
                    crowded |= short(device).map_err(phase("standby"))?;
                }
                // Collective transports suspend on every rank or on none.
                let group = self.world.union.as_ref().unwrap_or(&self.world.host);
                group
                    .all_gather_bytes(&[u8::from(crowded)])
                    .map_err(phase("standby"))?
                    .iter()
                    .any(|vote| vote.first() == Some(&1))
            }
        };
        if !crowded {
            tracing::info!(bytes, "the standby worker keeps its graph memory resident");
            return Ok(Suspension::default());
        }

        // Warmup ended: no work runs on the streams once they drained.
        for stream in self.streams.iter() {
            stream.synchronize().map_err(phase("standby"))?;
        }
        for pool in &pools {
            pool.suspend().map_err(phase("standby"))?;
        }
        tracing::info!(bytes, "the standby worker suspended its graph memory");
        // The expert transports release their workspaces: those that keep their
        // addresses are suspended, which the promotion maps back; the others
        // are freed, and the promotion rebuilds them and captures the
        // computations' graphs that address them again.
        let experts = match (
            self.executor.backend().experts(),
            self.executor.backend().executor(),
        ) {
            (Some(_), Some(executor)) => {
                let (suspended, released) = executor.park_experts().map_err(phase("standby"))?;
                if suspended {
                    tracing::info!("the standby worker suspended its expert transports");
                }
                if released {
                    tracing::info!(
                        "the standby worker released its expert workspaces, which its promotion rebuilds"
                    );
                }
                ParkedExperts {
                    suspended,
                    released,
                }
            }
            _ => ParkedExperts::default(),
        };
        // NCCL releases its communicators' buffers, the stream groups'
        // among them; the graphs that captured their collectives replay once
        // they resume.
        let world = self.world.torch().clone();
        Interpreter::import("uniserve.distributed.world.suspend")
            .and_then(|suspend| suspend.invoke(&[world.view()]))
            .map_err(phase("standby"))?;
        tracing::info!("the standby worker suspended its communicators");
        for &device in &devices {
            if short(device).map_err(phase("standby"))? {
                let memory = device
                    .memory_info()
                    .map_err(|error| phase("standby")(error.into()))?;
                return Err(phase("standby")(Error::Invalid(format!(
                    "the standby worker does not fit beside the workers it stands by for on \
                     {device}: {} bytes are free once its graph memory is suspended, below \
                     the {} bytes of graph memory its promotion maps",
                    memory.free,
                    graph_storage_budget_bytes(memory.total as i64)
                ))));
            }
        }
        Ok(Suspension {
            pools,
            experts,
            world: Some(world),
        })
    }

    /// Release every owner in dependency order. An aborted close never
    /// waits for device work or peers: it abandons what may still be in use
    /// until the process exits.
    pub(crate) fn close(mut self, aborted: bool) -> std::result::Result<(), WorkerError> {
        let mut failure: Option<WorkerError> = None;
        let mut record = |result: std::result::Result<(), WorkerError>| {
            if let Err(error) = result {
                match &mut failure {
                    Some(first) => first.note(&error),
                    None => failure = Some(error),
                }
            }
        };

        if !aborted {
            // The builder thread first performs every job still queued (work
            // nobody waited for, like a lifecycle command's), so that no job
            // addresses an owner while it closes.
            let drained = self.executor.backend().job_queue().drain(BLOCK_ON_TIMEOUT);
            record(drained.map_err(WorkerError::from));
            for stream in self.streams.iter() {
                record(stream.synchronize().map_err(WorkerError::from));
            }
        }
        let mut executor = self.executor;
        executor.backend_mut().close();
        if !aborted {
            record(executor.close());
        }
        // The executor releases its computations' graphs and execution
        // contexts, then the computations their output storage.
        let closed = executor.backend_mut().close_executor(aborted);
        record(closed.map_err(WorkerError::from));
        // Transports retire after every context and graph using them;
        // their close is collective over the expert union.
        let closed = executor.backend_mut().close_experts(aborted);
        record(closed.map_err(WorkerError::from));
        let closed = executor.backend_mut().close_storage(aborted);
        record(closed.map_err(WorkerError::from));
        if aborted {
            // Batches may still reference device work.
            crate::leak(executor);
        } else {
            drop(executor);
        }
        self.graphs.close();
        if !aborted {
            // The owners above gave up their Python objects without waiting
            // for the releases; an object may use its stream while it is
            // released, so the releases are made before the streams close.
            Interpreter::synchronize();
        }
        record(self.streams.close(aborted).map_err(WorkerError::from));
        if !aborted {
            // Every stream that wrote the exclusive storages is drained: a
            // successor takes them without a shoot-down.
            record(self.retained.release().map_err(WorkerError::from));
        }
        let mut world = self.world;
        record(world.close(aborted).map_err(WorkerError::from));
        drop(world);
        if aborted {
            crate::leak(self.model);
            crate::leak(self.context);
        } else {
            // The model is released before the context its tensors live in.
            drop(self.model);
            Interpreter::synchronize();
            drop(self.context);
            // Every owner has drained its streams; deferred frees can run.
            record(cuda::reclaim::flush().map_err(|error| WorkerError::from(Error::from(error))));
        }
        failure.map_or(Ok(()), Err)
    }

    /// Each CUDA device's graph pools may hold their current bytes plus the
    /// grant left after product storage not yet resident.
    fn bind_graph_budgets(&mut self) -> Result<()> {
        let Some(budget) = &self.budget else {
            return Ok(());
        };
        let devices: Vec<Device> = self.config.devices().collect();
        for device in devices {
            let Device::Cuda(index) = device else {
                continue;
            };
            // Product storage lives on the primary device only.
            let products = if device == self.config.device {
                self.plan.arena.deferred_device_bytes()
            } else {
                0
            };
            let measured = budget.budget(index, self.config.mem_fraction_static)?;
            let segments = self.segments(index)?;
            let pooled = self
                .graphs
                .pool_bytes(&segments)
                .get(&index)
                .copied()
                .unwrap_or(0);
            let amount = pooled + measured.available.saturating_sub(products);
            self.graphs
                .set_budget(index, amount, measured.process, pooled);
        }
        Ok(())
    }

    /// Bytes the caching allocator holds in each of its pools on `device`.
    fn segments(&self, device: DeviceId) -> Result<Vec<Segment>> {
        let function = "uniserve.cuda_graph.pool_bytes";
        let pools =
            Interpreter::import(function)?.call_values(&[Any::from(i64::from(device.0))])?;
        let pools =
            crate::ffi::integers(pools).map_err(|error| Error::numerical(function, &error))?;
        Ok(pools
            .as_chunks::<3>()
            .0
            .iter()
            .map(|&[first, second, bytes]| (device, (first as u64, second as u64), bytes as u64))
            .collect())
    }

    fn require_codecs(&self) -> std::result::Result<(), WorkerError> {
        Interpreter::import("uniserve.media.codec.require_codecs")
            .and_then(|function| function.invoke(&[]))
            .map_err(phase("warmup"))
    }

    /// Check graph residency, report memory and kernels, then seal.
    fn complete_startup(&mut self) -> std::result::Result<(), WorkerError> {
        let cuda_devices: Vec<DeviceId> = self.config.devices().filter_map(Device::cuda).collect();
        if let Some(budget) = &self.budget {
            let mut segments = Vec::new();
            for &device in &cuda_devices {
                segments.extend(self.segments(device).map_err(phase("warmup"))?);
            }
            let pools = self.graphs.pool_bytes(&segments);
            let mut failure = None;
            let residency = self.graphs.resident_bytes(&pools, |device| {
                budget.process_bytes(device).unwrap_or_else(|error| {
                    failure.get_or_insert(error);
                    0
                })
            });
            if let Some(error) = failure {
                return Err(phase("warmup")(error));
            }
            self.graphs.check(&residency).map_err(phase("warmup"))?;
            for &device in &cuda_devices {
                let process = budget.process_bytes(device).map_err(phase("warmup"))?;
                let reserved: i64 = Interpreter::import("torch.cuda.memory_reserved")
                    .and_then(|reserved| {
                        reserved.call_as(&[(&Any::from(i64::from(device.0))).into()])
                    })
                    .map_err(phase("warmup"))?;
                let gib = |bytes: u64| bytes as f64 / f64::from(1u32 << 30);
                tracing::info!(
                    "device storage on {device}: {:.2} GiB held by this process at startup, {:.2} GiB of it reserved by the caching allocator and {:.2} GiB of that in graph pools",
                    gib(process),
                    gib(reserved as u64),
                    gib(pools.get(&device).copied().unwrap_or(0)),
                );
            }
        }

        let executor = &mut self.executor;
        let mut records = KernelRecords::new().map_err(phase("warmup"))?;
        if let Some(python) = executor.backend().executor() {
            for (label, kernels) in python.kernels().map_err(phase("warmup"))? {
                records.add(&label, &self.config.device.to_string(), &kernels);
            }
        }
        records
            .log("startup", self.config.rank, &self.config.device.to_string())
            .map_err(phase("warmup"))?;
        // Every graph the executor captured is resident: it refuses
        // further captures.
        if let Some(python) = executor.backend().executor() {
            python.seal().map_err(phase("warmup"))?;
        }
        let kernel_choices =
            Interpreter::import("uniserve.backends.kernel_choices").map_err(phase("warmup"))?;
        let choices: i64 = kernel_choices.call_as(&[]).map_err(phase("warmup"))?;
        executor.backend_mut().watch_kernels(KernelWatch::new(
            kernel_choices,
            choices,
            records,
            self.config.rank,
            self.config.device.to_string(),
        ));
        self.graphs.seal();
        Ok(())
    }
}

/// Allocate every plugin's communication buffer on the rank's CUDA device
/// (`uniserve.plugins.allocate_buffers`), and measure what NVML attributes
/// to no process of what they took: the device's free memory falls by more
/// than the process's grows. Returns the objects that hold the buffers,
/// which the rank keeps, and those bytes by device; nothing for a CPU rank
/// or without buffers.
fn allocate_plugin_buffers(device: Device) -> Result<(Option<Handle>, BTreeMap<DeviceId, u64>)> {
    let Device::Cuda(index) = device else {
        return Ok((None, BTreeMap::new()));
    };
    let nvml = cuda::Nvml::init()?;
    let free = || -> Result<u64> { Ok(index.memory_info()?.free) };
    let process = || -> Result<u64> {
        nvml.process_memory(index).map_err(|error| match error {
            cuda::Error::ProcessNotFound { .. } => Error::Unsupported(format!(
                "NVML reports no device storage for this process on {index}"
            )),
            error => Error::Cuda(error),
        })
    };
    let before = (free()?, process()?);
    let held = Handle::new(
        Interpreter::import("uniserve.plugins.allocate_buffers")?
            .call_values(&[Any::from(i64::from(index.0))])?,
    )?;
    let count: usize = Interpreter::import("builtins.len")?.call_as(&[held.view()])?;
    if count == 0 {
        return Ok((None, BTreeMap::new()));
    }
    let after = (free()?, process()?);
    let taken = before.0.saturating_sub(after.0);
    let unattributed = taken.saturating_sub(after.1.saturating_sub(before.1));
    tracing::info!(
        buffers = count,
        bytes = taken,
        unattributed,
        "allocated the plugins' communication buffers"
    );
    Ok((Some(held), BTreeMap::from([(index, unattributed)])))
}

/// The model fingerprints of a rank that holds CUDA weights
/// (`uniserve.loading.retained.fingerprints`), which key its retained
/// weights and its cache pages; `None` for the stub model and a rank
/// without CUDA weights.
fn model_fingerprints(
    launch: &WorkerLaunch,
    description: &crate::worker::loading::Description,
    host_only: bool,
) -> Result<Option<weights::Fingerprints>> {
    if host_only || launch.device().cuda().is_none() {
        return Ok(None);
    }
    let started = std::time::Instant::now();
    let fingerprints = LoadedModel::fingerprint(description, launch)?;
    tracing::debug!(
        elapsed_ms = started.elapsed().as_millis() as u64,
        "fingerprinted the model's weights"
    );
    Ok(fingerprints)
}

/// What every rank in `agreement` reads from the checkpoint, from each
/// rank's own `need` (`weights::Load`).
fn agree_load(
    agreement: &crate::parallel::host::HostGroup,
    need: weights::Load,
) -> Result<weights::Load> {
    let encoded = serde_json::to_vec(&need)
        .map_err(|error| Error::Invalid(format!("encoding a weight load: {error}")))?;
    let needs = agreement
        .all_gather_bytes(&encoded)?
        .iter()
        .map(|bytes| {
            serde_json::from_slice(bytes)
                .map_err(|error| Error::Interface(format!("a rank's weight load: {error}")))
        })
        .collect::<Result<Vec<weights::Load>>>()?;
    Ok(weights::Load::union(needs))
}

/// The keys of the rank's retained weights, or `None` when its weights are
/// not retained: a rank without a model fingerprint, and a rank whose
/// expert weights move to memory of the expert union after loading
/// (`dwdp`).
fn weight_keys(
    launch: &WorkerLaunch,
    fingerprints: Option<&weights::Fingerprints>,
    retained: &Retained,
) -> Result<Option<WeightKeys>> {
    let moved = launch
        .expert_parallel
        .as_ref()
        .is_some_and(ExpertUnion::prefetches_weights);
    let Some(fingerprints) = fingerprints.filter(|_| !moved) else {
        return Ok(None);
    };
    let devices: Vec<DeviceId> = launch
        .execution
        .devices()
        .filter_map(|device| device.cuda())
        .collect();
    Ok(Some(WeightKeys::new(
        retained,
        fingerprints,
        &weights::placement(launch)?,
        &devices,
    )?))
}

/// The worker's identity for transfers: the deployment's name of its host,
/// its process and its incarnation. The engine knows which worker group and
/// rank the incarnation serves.
fn endpoint(host: &str) -> WorkerEndpoint {
    WorkerEndpoint {
        host: host.to_owned(),
        address_space: format!("{}:{}", std::process::id(), uuid::Uuid::new_v4().simple()),
        incarnation: uuid::Uuid::new_v4().simple().to_string(),
    }
}

/// The graph policy a rank serves with.
///
/// `auto` captures CUDA graphs when serving captures the loaded attention
/// provider's calls (`uniserve.backends.attention.Backend.captures`) and
/// otherwise serves every call eagerly; `full` requires that capture; `off`
/// never captures. Every rank of a group loads the same provider, so the
/// group resolves one policy and captures the same shapes.
///
/// A rank holding `host_component`, a component that keeps its weights in
/// host memory, requires `off`: its layers' weight copies are scheduled by
/// the host as each layer is reached, which a captured graph cannot repeat.
fn graph_policy(
    configured: &str,
    model: &LoadedModel,
    host_component: Option<&str>,
) -> Result<String> {
    if let Some(name) = host_component
        && configured != "off"
    {
        return Err(Error::Unsupported(format!(
            "component {name} keeps its weights in host memory, which requires cuda_graph_config.policy=off"
        )));
    }
    if model.attention_captures || configured == "off" {
        return Ok(configured.to_owned());
    }
    if configured == "full" {
        return Err(Error::Unsupported(format!(
            "graph policy full requires an attention provider whose calls serving captures; the {} provider's are not",
            model.attention_backend
        )));
    }
    tracing::info!(
        "graph policy auto serves every call eagerly: the {} attention provider's calls are not captured",
        model.attention_backend
    );
    Ok("off".to_owned())
}

/// A rank's graph mode and the buckets each execution path captures, as a
/// plugin's graph policy reads and chooses them
/// (`uniserve.plugins.Registry.graph_policy`).
#[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphChoice {
    mode: String,
    decode: Vec<usize>,
    prefill: Vec<usize>,
    flow_shapes: Vec<[usize; 2]>,
}

impl GraphChoice {
    /// The configured mode and buckets of `config`.
    fn of(config: &WorkerConfig) -> Self {
        Self {
            mode: config.graph_policy.clone(),
            decode: config.decode_graph_batch_sizes.clone(),
            prefill: config.prefill_graph_token_sizes.clone(),
            flow_shapes: config
                .flow_graph_shapes
                .iter()
                .map(|&(height, width)| [height, width])
                .collect(),
        }
    }

    /// Checks a choice as the daemon checks the configuration: a known
    /// mode, positive and strictly increasing sizes, positive and distinct
    /// shapes.
    fn check(&self) -> Result<()> {
        let invalid =
            |message: String| Err(Error::Invalid(format!("a plugin's graph policy {message}")));
        if !["auto", "full", "off"].contains(&self.mode.as_str()) {
            return invalid(format!("chooses the unknown mode {:?}", self.mode));
        }
        for (name, sizes) in [("decode", &self.decode), ("prefill", &self.prefill)] {
            if sizes.first() == Some(&0) || !sizes.windows(2).all(|pair| pair[0] < pair[1]) {
                return invalid(format!(
                    "chooses {name} sizes that are not positive and strictly increasing: {sizes:?}"
                ));
            }
        }
        let shapes: std::collections::BTreeSet<_> = self.flow_shapes.iter().collect();
        if shapes.len() != self.flow_shapes.len() || shapes.iter().any(|shape| shape.contains(&0)) {
            return invalid(format!(
                "chooses flow shapes that are not positive and distinct: {:?}",
                self.flow_shapes
            ));
        }
        Ok(())
    }

    /// States the choice in `config`.
    fn apply(self, config: &mut WorkerConfig) {
        config.graph_policy = self.mode;
        config.decode_graph_batch_sizes = self.decode;
        config.prefill_graph_token_sizes = self.prefill;
        config.flow_graph_shapes = self
            .flow_shapes
            .into_iter()
            .map(|[height, width]| (height, width))
            .collect();
    }
}

/// Lets a plugin choose the rank's graph mode and capture buckets
/// (`uniserve.plugins.graph_policy`) from the loaded models and the
/// configured ones, before the rank resolves its policy; the configured
/// ones stand without such a plugin.
fn plugin_graph_policy(config: &mut WorkerConfig) -> Result<()> {
    let configured = GraphChoice::of(config);
    let function = "uniserve.plugins.graph_policy";
    let configured_json = serde_json::to_string(&configured)
        .map_err(|error| Error::Invalid(format!("graph policy: {error}")))?;
    let chosen: tvm_ffi::String = Interpreter::import(function)?
        .call_values(&[Any::from(tvm_ffi::String::from(configured_json))])?
        .try_into()
        .map_err(|error| Error::numerical(function, &error))?;
    if chosen.as_str().is_empty() {
        return Ok(());
    }
    let chosen: GraphChoice = serde_json::from_str(chosen.as_str())
        .map_err(|error| Error::Invalid(format!("a plugin's graph policy: {error}")))?;
    chosen.check()?;
    if chosen != configured {
        tracing::info!(?chosen, "a plugin chose the graph policy");
    }
    chosen.apply(config);
    Ok(())
}

/// Whether a rank of `role` leaves calls of `kind` to the groups of the
/// other role: a prefill group's requests decode on a decode group, which
/// installs the KV cache the prefill exported (`WorkerGroupRole`).
fn left_to_other_role(role: &str, kind: CallKind) -> bool {
    match role {
        "prefill" => matches!(
            kind,
            CallKind::Forward(ipc::ForwardMode::Decode | ipc::ForwardMode::Verify)
        ),
        "decode" => kind == CallKind::Forward(ipc::ForwardMode::Prefill),
        _ => false,
    }
}

/// Whether the rank publishes device products over CUDA VMM.
fn exports_device_products(launch: &WorkerLaunch) -> bool {
    launch
        .data_plane
        .exports
        .iter()
        .any(|name| name == TransportKind::CudaVmm.name())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The configured choice states the configuration unchanged, and the
    /// check refuses what the daemon refuses in the configuration.
    #[test]
    fn a_graph_choice_is_checked_as_the_configuration() -> Result<()> {
        let config = crate::capacity::plan::tests::cpu_config();
        let configured = GraphChoice::of(&config);
        configured.check()?;
        let mut stated = config.clone();
        GraphChoice::of(&config).apply(&mut stated);
        assert_eq!(GraphChoice::of(&stated), configured);

        let refused = [
            GraphChoice {
                mode: "eager".to_owned(),
                ..GraphChoice::of(&config)
            },
            GraphChoice {
                decode: vec![1, 4, 4],
                ..GraphChoice::of(&config)
            },
            GraphChoice {
                prefill: vec![0, 16],
                ..GraphChoice::of(&config)
            },
            GraphChoice {
                flow_shapes: vec![[1024, 1024], [1024, 1024]],
                ..GraphChoice::of(&config)
            },
            GraphChoice {
                flow_shapes: vec![[0, 1024]],
                ..GraphChoice::of(&config)
            },
        ];
        for choice in refused {
            assert!(choice.check().is_err(), "{choice:?}");
        }
        Ok(())
    }
}

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
//! 4. Loading, model facts, execution streams.
//! 5. Capacity: page size, request slots, cache units and latent pages, all
//!    agreed over the group; the auxiliary-device check.
//! 6. Request storage and the runners, created by the storage and
//!    capability runtimes on the plan and handed to the backend.
//! 7. The executor and the profiler.
//!
//! [`Worker::warmup`] runs the warmup phases ([`crate::warmup`]),
//! [`Worker::run`] serves the channel, and [`Worker::close`] releases owners
//! in dependency order: execution, runners, streams, requests, the process
//! world, then deferred device memory.

mod loading;
mod storage;

pub(crate) use loading::LoadedModel;
pub use storage::RankStorage;

use std::collections::BTreeMap;

use std::sync::Arc;
use tvm_ffi::Any;
use uniserve_core::ComponentConfig;

use uniserve_cuda::{DeviceId, HandleType, PrimaryContext};
use uniserve_ipc::{CallKind, Channel, ComponentInfo, ForwardMode, WorkerEndpoint, WorkerInfo};

use crate::capacity::checks;
use crate::capacity::device::{DeviceBudget, total_bytes};
use crate::capacity::plan::{LatentPlan, Plan, Report, TokenFacts, TokenLayout, page_size};
use crate::capacity::video::VideoCapacity;
use crate::config::placement::Capability;
use crate::config::placement::Components;
use crate::config::{Device, WorkerConfig, WorkerLaunch};
use crate::cuda_graph::reserve_pools;
use crate::cuda_graph::{GraphStorage, graph_storage_budget_bytes};
use crate::error::WorkerError;
use crate::executor::Executor;
use crate::executor::backend::WorkerBackend;
use crate::executor::forward::transfer::TransferRuntime;
use crate::ffi::Device as StorageDevice;
use crate::ffi::{DType, Handle, Interpreter};
use crate::lane::BLOCK_ON_TIMEOUT;
use crate::parallel::world::{World, startup_timeout};
use crate::parallel::{DwdpManager, ExpertStart, ExpertUnion, ExpertWarmup, PeerGroup};
use crate::profiling::kernels::{KernelRecords, KernelWatch};
use crate::profiling::profiler::{ProfileConfig, Profiler};
use crate::profiling::range;
use crate::request::video::VideoRequests;
use crate::runner::causal_lm::CausalLMRunners;
use crate::runner::causal_lm::start::{Startup, TextBounds, expert_tokens, reserved_bytes};
use crate::runner::causal_lm::warmup::CausalLMWarmup;
use crate::runner::component::VideoSetup;
use crate::runner::decoder::warmup::DecoderWarmup;
use crate::runner::denoiser::image_model::ImageModel;
use crate::runner::denoiser::warmup::DenoiserWarmup;
use crate::runner::denoiser::{VideoModel, placed_denoiser};
use crate::runner::diffusion_lm::{self, DiffusionLMRunner, DiffusionLMWarmup};
use crate::runner::encoder::EncoderInfo;
use crate::runner::encoder::warmup::EncoderWarmup;
use crate::runner::scratch::Scratches;
use crate::runner::start::{
    ImageOwners, Startup as ImageStartup, reserved_bytes as image_reserved_bytes, start_video,
};
use crate::service::Service;
use crate::storage::{BufferPool, CachePlan, GroupShape, PrefixCache, TensorStore};
use crate::stream::{CUDAStream, EventPool, Streams};
use crate::transfer::{Capacity, TransportKind, TransportOptions, Transports};
use crate::warmup::{Scenarios, WarmupHooks};
use crate::{Error, Result};
use loading::{ComponentMesh, Description, ExpertPartition, sequence_kind};

/// One caching-allocator segment: device, graph pool id and bytes.
type Segment = (i32, (u64, u64), u64);

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
pub struct Worker {
    config: WorkerConfig,
    info: WorkerInfo,
    plan: Plan,
    model: Option<LoadedModel>,
    world: Option<World>,
    budget: Option<DeviceBudget>,
    streams: Streams,
    graphs: GraphStorage<String, Handle>,
    /// The shared work areas of every stream runners construct their
    /// execution contexts on.
    _scratches: Scratches,
    executor: Option<Executor<WorkerBackend>>,
    hooks: Vec<Box<dyn WarmupHooks>>,
    /// Retained for the rank's lifetime: torch, the driver owners and the
    /// reclaimer all use the device's primary context.
    context: Option<PrimaryContext>,
    warmed: bool,
    closed: bool,
}

impl Worker {
    /// Build the rank. On failure, everything built so far is released;
    /// the process world is abandoned, because peers may still be inside a
    /// collective.
    pub fn start(launch: WorkerLaunch) -> std::result::Result<Self, WorkerError> {
        let device = launch.device();
        let context = match device {
            Device::Cuda(index) => {
                let count = DeviceId::count().map_err(|error| phase("device")(error.into()))?;
                if index >= count {
                    return Err(phase("device")(Error::Unsupported(format!(
                        "cuda device {device} is outside the {count} visible CUDA device(s)"
                    ))));
                }
                Some(
                    PrimaryContext::retain(DeviceId(index))
                        .map_err(|error| phase("device")(error.into()))?,
                )
            }
            Device::Cpu => None,
        };

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

        let mut config = launch.execution.clone();
        let components = Components::new(
            &description.declarations,
            &launch.components,
            config.rank,
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
        if description.video_decoder {
            config = config
                .for_media_slots(launch.ipc.queue_depth)
                .map_err(phase("load"))?;
        }
        let loading = range(c"uniserve.worker.load", None);
        let model = LoadedModel::load(description, &launch, meshes, experts, host_only)
            .map_err(phase("load"))?;
        let image = ImageModel::describe(model.handle()).map_err(phase("load"))?;
        let video = if model.video_decoder {
            let denoiser = placed_denoiser(&model.declarations, &config.deployment_components)
                .map_err(phase("load"))?;
            Some(VideoModel::new(model.handle(), denoiser).map_err(phase("load"))?)
        } else {
            None
        };
        drop(loading);

        let devices: Vec<Device> = config.devices().collect();
        let mut streams = Streams::create(&config, &devices).map_err(phase("streams"))?;
        let node = host_name().map_err(phase("host"))?;
        // Expert transports map their device workspaces before capacity
        // planning measures free memory.
        let mut experts = ExpertUnion::start(ExpertStart {
            launch: &launch,
            world: &world,
            model: &model,
            streams: &mut streams,
            max_tokens: expert_tokens(&config),
            node: &node,
        })
        .map_err(phase("experts"))?;
        let weights = DwdpManager::start(&launch, &world, &model, &streams, &node)
            .map_err(phase("experts"))?;
        let budget = match device {
            Device::Cuda(_) => Some(
                DeviceBudget::new(
                    Interpreter::import("torch.cuda.synchronize").map_err(phase("capacity"))?,
                    Interpreter::import("torch.cuda.empty_cache").map_err(phase("capacity"))?,
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
            &image,
            &world,
            budget.as_ref(),
            &components,
            streams.lanes().len().max(1),
            video.as_ref(),
        )
        .map_err(phase("capacity"))?;
        if let Some(budget) = &budget {
            checks::auxiliary(&config, &plan.fixed_device_bytes, budget)
                .map_err(phase("capacity"))?;
        }
        drop(planning);
        let info = plan.info.clone();
        info.validate()
            .map_err(|error| phase("capacity")(Error::Invariant(error.to_string())))?;
        log_capacity(&config, &plan);

        let budgets = devices
            .iter()
            .filter_map(|device| device.cuda_index())
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

        let storage =
            allocate_storage(&launch, &config, &plan, &streams).map_err(phase("storage"))?;
        let mut components = components;
        let preparation = device
            .cuda_index()
            .and_then(|index| streams.preparation(DeviceId(index)));
        let preparation = preparation
            .map(CUDAStream::stream)
            .transpose()
            .map_err(phase("bind"))?;
        // Numerical collectives of each runner run on NCCL communicators
        // bound to its stream, over its component's mesh groups.
        let first = launch
            .expert_parallel
            .as_ref()
            .map_or(0, |union| union.rank - launch.execution.rank);
        let peers = PeerGroup {
            group: &world.host,
            collectives: launch
                .components
                .keys()
                .map(|name| (name.clone(), world.collective_groups(name, first)))
                .collect(),
        };
        let text = CausalLMRunners::start(Startup {
            config: &config,
            request_slots: info.request_slots as usize,
            queue_depth: launch.ipc.queue_depth,
            max_payload_bytes: launch.ipc.max_payload_bytes,
            model: &model,
            image_processor: image.processing.is_some(),
            streams: &streams,
            prefix_cache: storage.prefix_cache.as_ref(),
            components: &components,
            scratches: &mut scratches,
            graphs: &mut graphs,
            preparation,
            experts: experts.as_mut(),
            peers: Some(&peers),
            weights: weights.as_ref(),
        })
        .map_err(phase("bind"))?;
        // A video model's encoders and denoiser run as media batches.
        let image = if video.is_some() {
            None
        } else {
            ImageOwners::start(ImageStartup {
                config: &config,
                request_slots: info.request_slots as usize,
                queue_depth: launch.ipc.queue_depth,
                host_capacity: info.host_lane_capacity as usize,
                model: &model,
                streams: &streams,
                storage: &storage,
                latent: plan.latent,
                components: &components,
                scratches: &mut scratches,
                graphs: &mut graphs,
                preparation,
            })
            .map_err(phase("bind"))?
        };
        let diffusion_lm = DiffusionLMRunner::start(diffusion_lm::Startup {
            config: &config,
            request_slots: info.request_slots as usize,
            queue_depth: launch.ipc.queue_depth,
            max_payload_bytes: launch.ipc.max_payload_bytes,
            model: &model,
            experts: expert_exchange(&launch),
            streams: &streams,
            prefix_cache: storage.prefix_cache.as_ref(),
            components: &mut components,
            scratches: &mut scratches,
            graphs: &mut graphs,
            preparation,
            peers: Some(&peers),
            expert_runtime: experts.as_mut(),
            weights: weights.as_ref(),
        })
        .map_err(phase("bind"))?;
        if let Some(experts) = experts.as_mut() {
            if dedicated {
                experts
                    .bind_experts(&model, &mut scratches)
                    .map_err(phase("bind"))?;
            }
            reserve_join_pools(&config, experts, &mut graphs).map_err(phase("bind"))?;
        }
        let mut streams = streams;
        let video_owners = match (video, video_capacity) {
            (Some(video), Some(capacity)) => {
                let (encoders, denoisers, decoders, media) = start_video(
                    VideoSetup {
                        loaded: model.handle(),
                        model: &video,
                        capacity: &capacity,
                        info: &info,
                        config: &config,
                        streams: &mut streams,
                        graphs: &mut graphs,
                        attention: model.attention_backend.clone(),
                    },
                    &components,
                )
                .map_err(phase("storage"))?;
                let device = match config.device {
                    crate::config::Device::Cuda(index) => Some(index),
                    crate::config::Device::Cpu => None,
                };
                let leases = 2 * info.queue_depth as usize + 2;
                let words = info.max_batch_calls.max(1) as usize;
                Some((
                    VideoRequests::new(video, capacity.media).map_err(phase("storage"))?,
                    encoders,
                    denoisers,
                    decoders,
                    media,
                    crate::OutputPool::new(device, leases, words),
                ))
            }
            _ => None,
        };
        let mut backend =
            WorkerBackend::new(info.clone(), config.rank, components).map_err(phase("storage"))?;
        backend.set_rank_storage(storage);
        backend.set_job_queues(streams.job_queues());
        let text_warmup = text.is_some();
        let video_warmup = video_owners.is_some();
        if let Some((requests, encoders, denoisers, decoders, media, outputs)) = video_owners {
            backend.set_video(requests, encoders, denoisers, decoders, media, outputs);
        }
        if let Some(text) = text {
            backend.set_text(text);
        }
        let image_warmup = image.is_some();
        if let Some(image) = image {
            backend.set_image(image);
        }
        let diffusion_lm_warmup = !diffusion_lm.is_empty();
        if diffusion_lm_warmup {
            backend.set_diffusion_lm(diffusion_lm);
        }
        if let Some(weights) = weights {
            backend.set_weights(weights);
        }
        let expert_warmup = experts.is_some();
        if let Some(experts) = experts {
            backend.set_experts(experts);
        }
        let cuda: Vec<i32> = devices
            .iter()
            .filter_map(|device| device.cuda_index())
            .collect();
        backend.set_transfer(
            TransferRuntime::start(
                &streams,
                &cuda,
                &CallKind::ALL,
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
                    device.cuda_index(),
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
        let mut hooks: Vec<Box<dyn WarmupHooks>> = Vec::new();
        if video_warmup || image_warmup {
            // Prepared and captured in this order on every rank.
            hooks.push(Box::new(EncoderWarmup));
            hooks.push(Box::new(DenoiserWarmup));
            hooks.push(Box::new(DecoderWarmup));
        }
        if text_warmup {
            hooks.push(Box::new(CausalLMWarmup {
                row_tokens: config
                    .max_sequence_tokens
                    .min(config.max_batch_tokens)
                    .max(1),
            }));
        }
        if diffusion_lm_warmup {
            hooks.push(Box::new(DiffusionLMWarmup));
        }
        if expert_warmup {
            // After the text captures, whose eager warmups were announced
            // expert steps.
            hooks.push(Box::new(ExpertWarmup));
        }

        tracing::info!(target: "uniserve_worker.launch", "loaded numerical model {}", model.model_name);
        Ok(Self {
            config,
            info,
            plan,
            model: Some(model),
            world: Some(world),
            budget,
            streams,
            graphs,
            _scratches: scratches,
            executor: Some(executor),
            hooks,
            context,
            warmed: false,
            closed: false,
        })
    }

    pub fn info(&self) -> &WorkerInfo {
        &self.info
    }

    pub fn config(&self) -> &WorkerConfig {
        &self.config
    }

    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    pub fn world(&self) -> Option<&World> {
        self.world.as_ref()
    }

    pub fn streams_mut(&mut self) -> &mut Streams {
        &mut self.streams
    }

    pub fn backend_mut(&mut self) -> Option<&mut WorkerBackend> {
        self.executor.as_mut().map(Executor::backend_mut)
    }

    /// Register a capability runtime's warmup work.
    pub fn add_warmup(&mut self, hooks: Box<dyn WarmupHooks>) {
        self.hooks.push(hooks);
    }

    /// Run warmup once, before the first serving submission.
    pub fn warmup(&mut self) -> std::result::Result<(), WorkerError> {
        if self.warmed {
            return Ok(());
        }
        let executor = self
            .executor
            .as_mut()
            .ok_or_else(|| WorkerError::from(Error::State("the worker is closed")))?;
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
        let executor = self
            .executor
            .as_mut()
            .ok_or_else(|| WorkerError::from(Error::State("the worker is closed")))?;
        for hooks in &mut self.hooks {
            hooks.prepare(executor.backend_mut())?;
        }
        {
            let _capture = range(c"uniserve.worker.capture", None);
            for hooks in &mut self.hooks {
                hooks.capture(executor.backend_mut())?;
            }
        }
        {
            let mut scenarios = Scenarios::new(executor);
            scenarios.lifecycle()?;
            if cuda && !scenarios.backend().requests().has_open_requests()? {
                for hooks in &mut self.hooks {
                    hooks.scenarios(&mut scenarios)?;
                }
            }
        }
        let backend = executor.backend();
        if backend.components().holds_host() {
            self.require_codecs()?;
        }
        self.complete_startup()?;

        let executor = self
            .executor
            .as_mut()
            .ok_or_else(|| WorkerError::from(Error::State("the worker is closed")))?;
        if !executor.backend().requests().request_ids().is_empty() {
            return Err(Error::State("startup completed with resident requests").into());
        }
        // Synthetic batches consumed ids and collective sequences; serving
        // starts from the engine's first batch.
        executor.reset()?;
        if let Some(budget) = &self.budget {
            checks::startup(
                &self.config,
                self.plan.arena.device_product_bytes,
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

    /// Serve the channel until the engine closes the rank or a fatal
    /// response ends admission.
    pub fn run(&mut self, channel: &mut Channel) -> std::result::Result<(), WorkerError> {
        self.warmup()?;
        let info = self.info.clone();
        let executor = self
            .executor
            .as_mut()
            .ok_or_else(|| WorkerError::from(Error::State("the worker is closed")))?;
        // Ranks of an expert union join every step outside their own.
        let experts = executor.backend().experts().is_some();
        executor.backend().set_storage_wake(Some(channel.wake()));
        let served = Service::new(executor, channel, info, experts).run();
        executor.backend().set_storage_wake(None);
        served
    }

    /// Release every owner in dependency order. An aborted close never
    /// waits for device work or peers: it abandons what may still be in use
    /// until the process exits.
    pub fn close(&mut self, aborted: bool) -> std::result::Result<(), WorkerError> {
        if std::mem::replace(&mut self.closed, true) {
            return Ok(());
        }
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
            // The lanes first perform what is still queued on them (work
            // nobody waited for, like a lifecycle command's), so that no
            // job addresses an owner while it closes.
            if let Some(executor) = &self.executor {
                let drained = executor.backend().job_queues().drain(BLOCK_ON_TIMEOUT);
                record(drained.map_err(WorkerError::from));
            }
            for stream in self.streams.iter() {
                record(stream.synchronize().map_err(WorkerError::from));
            }
        }
        if let Some(mut executor) = self.executor.take() {
            executor.backend_mut().close();
            if !aborted {
                record(executor.close());
            }
            // Each runner releases its graphs, then its execution context.
            let closed = executor.backend_mut().close_runtimes(aborted);
            record(closed.map_err(WorkerError::from));
            // Transports retire after every context and graph using them;
            // their close is collective over the expert union.
            let closed = executor.backend_mut().close_experts(aborted);
            record(closed.map_err(WorkerError::from));
            let closed = executor.backend_mut().close_storage(aborted);
            record(closed.map_err(WorkerError::from));
            if aborted {
                // Batches may still reference device work.
                #[allow(clippy::mem_forget)]
                std::mem::forget(executor);
            }
        }
        self.graphs.close();
        if !aborted {
            // The owners above gave up their Python objects without waiting
            // for the releases; an object may use its stream while it is
            // released, so the releases are made before the streams close.
            Interpreter::synchronize();
        }
        record(self.streams.close(aborted).map_err(WorkerError::from));
        if let Some(mut world) = self.world.take() {
            record(world.close(aborted).map_err(WorkerError::from));
        }
        if aborted {
            #[allow(clippy::mem_forget)]
            std::mem::forget(self.model.take());
            #[allow(clippy::mem_forget)]
            std::mem::forget(self.context.take());
        } else {
            // The model is released before the context its tensors live in.
            self.model = None;
            Interpreter::synchronize();
            self.context = None;
            // Every owner has drained its streams; deferred frees can run.
            record(
                uniserve_cuda::reclaim::flush()
                    .map_err(|error| WorkerError::from(Error::from(error))),
            );
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
                self.plan.arena.device_product_bytes
            } else {
                0
            };
            let measured = budget.budget(index, self.config.cache_storage_fraction)?;
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
    fn segments(&self, device: i32) -> Result<Vec<Segment>> {
        let function = "uniserve.cuda_graph.pool_bytes";
        let pools = Interpreter::import(function)?.call_values(&[Any::from(i64::from(device))])?;
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
        let cuda_devices: Vec<i32> = self
            .config
            .devices()
            .filter_map(Device::cuda_index)
            .collect();
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
                        reserved.call_as(&[(&Any::from(i64::from(device))).into()])
                    })
                    .map_err(phase("warmup"))?;
                let gib = |bytes: u64| bytes as f64 / f64::from(1u32 << 30);
                tracing::info!(
                    target: crate::profiling::kernels::TARGET,
                    "device storage on cuda:{device}: {:.2} GiB held by this process at startup, {:.2} GiB of it reserved by the caching allocator and {:.2} GiB of that in graph pools",
                    gib(process),
                    gib(reserved as u64),
                    gib(pools.get(&device).copied().unwrap_or(0)),
                );
            }
        }

        let executor = self
            .executor
            .as_mut()
            .ok_or_else(|| WorkerError::from(Error::State("the worker is closed")))?;
        let mut records = KernelRecords::new().map_err(phase("warmup"))?;
        for (label, device, kernels) in executor.backend().kernels().map_err(phase("warmup"))? {
            records.add(&label, &device, &kernels);
        }
        if executor.backend().serves_video() {
            let backend = executor.backend();
            let mut video = backend.encoders().text_kernels().map_err(phase("warmup"))?;
            video.extend(backend.denoisers().kernels().map_err(phase("warmup"))?);
            video.extend(backend.decoders().kernels().map_err(phase("warmup"))?);
            video.extend(
                backend
                    .encoders()
                    .latent_kernels()
                    .map_err(phase("warmup"))?,
            );
            for (label, kernels) in video {
                records.add(&label, &self.config.device.to_string(), &kernels);
            }
        }
        for runner in executor.backend_mut().diffusion_lm_mut() {
            let (label, kernels) = runner.kernels().map_err(phase("warmup"))?;
            records.add(&label, &self.config.device.to_string(), &kernels);
            runner.seal();
        }
        records
            .log("startup", self.config.rank, &self.config.device.to_string())
            .map_err(phase("warmup"))?;
        if let Some(text) = executor.backend_mut().text_mut() {
            text.seal();
        }
        if let Some(image) = executor.backend_mut().parts().image {
            image.encoders.seal_image();
            image.denoisers.seal_image();
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

impl Drop for Worker {
    fn drop(&mut self) {
        // A worker dropped without a close failed while starting or serving:
        // its peers may be blocked, so nothing waits.
        let _ = self.close(true);
    }
}

/// Allocate the storage owners on the plan: device events, the product
/// transports and the prefix cache of a text model, whose planes are reset on
/// the device's preparation stream.
fn allocate_storage(
    launch: &WorkerLaunch,
    config: &WorkerConfig,
    plan: &Plan,
    streams: &Streams,
) -> Result<RankStorage> {
    let events = EventPool::new();
    let kinds = launch
        .data_plane
        .backends
        .iter()
        .map(|name| TransportKind::parse(name))
        .collect::<Result<Vec<_>>>()?;
    let capacity = Arc::new(Capacity::new(
        plan.arena.transfer_bytes,
        plan.arena.transfer_tickets,
        storage::notify,
    )?);
    let options = TransportOptions {
        source: plan.info.endpoint.clone(),
        acknowledgment_slot: launch.ipc.acknowledgment_slot as usize,
        host_slots: launch
            .ipc
            .host_slots
            .iter()
            .map(|&slot| slot as usize)
            .collect(),
        cross_host_consumers: launch.ipc.products_cross_hosts,
        vmm_pool_bytes: plan.arena.transfer_bytes,
    };
    let exports = launch
        .data_plane
        .exports
        .iter()
        .map(|name| TransportKind::parse(name))
        .collect::<Result<Vec<_>>>()?;
    let transports = Arc::new(
        Transports::new(&kinds, capacity, Arc::clone(&events), &options)?.with_exports(&exports)?,
    );

    // Products: scheduler-placed buffers in the buffer pool and scalar relay
    // lanes, one per request slot and unresolved call, on the primary
    // device where every binding computes.
    let devices = vec![match config.device {
        Device::Cuda(index) => StorageDevice::Cuda(index),
        Device::Cpu => StorageDevice::Cpu,
    }];
    let physical = plan.physical_buffer_pool_bytes.max(1);
    let buffer_pool =
        BufferPool::allocate(physical, physical < plan.info.buffer_pool_bytes, &devices)?;
    let tensors = TensorStore::new(
        plan.arena.tensor_store,
        plan.arena.device_product_bytes.max(1) as usize,
        plan.info.encoder_entry_bytes.max(1) as usize,
        devices,
        plan.info.request_slots as usize,
        plan.info.max_unresolved_calls as usize + 1,
        buffer_pool,
        Arc::clone(&events),
    )?;

    let device = config.device.cuda_index();
    let preparation = device.and_then(|index| streams.preparation(DeviceId(index)));
    let prefix_cache = match (&plan.cache, &plan.info.prefix_cache) {
        (Some(cache), Some(info)) => Some(PrefixCache::new(
            Arc::clone(cache),
            info.num_units,
            plan.info.request_slots,
            cache_width(&cache.group_shapes()?, config.max_sequence_tokens),
            plan.info.max_unresolved_calls as usize,
            device,
            preparation.map(CUDAStream::stream).transpose()?,
        )?),
        _ => None,
    };
    if let Some(stream) = preparation {
        stream.synchronize()?;
    }
    // Slot releases order after the token batches that read them.
    let release_stream = match device {
        Some(index) => streams
            .stream_for(CallKind::Forward(ForwardMode::Decode), DeviceId(index))
            .map(|stream| stream.stream().cloned())
            .transpose()?,
        None => None,
    };
    Ok(RankStorage::new(
        events,
        tensors,
        transports,
        prefix_cache,
        release_stream,
    ))
}

/// Block-table width that holds the longest admitted sequence in every
/// group.
fn cache_width(shapes: &[GroupShape], max_sequence_tokens: usize) -> u32 {
    shapes
        .iter()
        .map(|shape| shape.resident_width(max_sequence_tokens as u64))
        .max()
        .unwrap_or(1)
        .try_into()
        .unwrap_or(u32::MAX)
}

/// Report the planned capacity once, for operators.
fn log_capacity(config: &WorkerConfig, plan: &Plan) {
    let info = &plan.info;
    let prefix_cache = info.prefix_cache.as_ref().map_or_else(
        || "no cache pool".to_owned(),
        |cache| {
            format!(
                "{} cache units of {} bytes in {} groups",
                cache.num_units,
                cache.unit_bytes,
                cache.groups.len()
            )
        },
    );
    tracing::info!(
        target: "uniserve_worker.launch",
        "planned capacity: {prefix_cache}, KV page size {} tokens, {} request slots, {} calls and {} tokens per batch, {} latent pages, {} buffer pool bytes",
        config.block_size.unwrap_or(0),
        info.request_slots,
        info.max_batch_calls,
        info.max_batch_tokens,
        info.latent_pages,
        info.buffer_pool_bytes,
    );
}

/// This host's node name, the rank's node identity in transfers.
///
/// Peers compare node names to decide whether a product location is on their
/// own host, so a substituted or altered name would misroute transfers: a
/// name that is empty or not UTF-8 is an error.
fn host_name() -> Result<String> {
    let system = rustix::system::uname();
    let node = system.nodename();
    match node.to_str() {
        Ok(name) if !name.is_empty() => Ok(name.to_owned()),
        _ => Err(Error::HostName(node.to_owned())),
    }
}

/// Graph pools for an expert union member's captured joins, one per microbatch,
/// when no text runner shares its pools on the microbatch streams.
fn reserve_join_pools(
    config: &WorkerConfig,
    experts: &mut ExpertUnion,
    graphs: &mut GraphStorage<String, Handle>,
) -> Result<()> {
    if config.graph_policy == "off" || experts.has_pools() {
        return Ok(());
    }
    let Device::Cuda(device) = config.device else {
        return Ok(());
    };
    let create = Interpreter::import("uniserve.cuda_graph.pool")?;
    let mut pools = Vec::new();
    for peer in 0..experts.microbatches() {
        let stream = experts.stream(peer).raw();
        let reserved = reserve_pools(
            graphs,
            &create,
            stream,
            format!("experts:join:{peer}"),
            &[device],
            None,
        )?;
        pools.extend(reserved.first().map(|pool| pool.value.clone()));
    }
    experts.set_pools(pools);
    Ok(())
}

/// The rank's identity for transfers: host, process and incarnation.
fn endpoint(worker_id: &str, rank: usize) -> Result<WorkerEndpoint> {
    Ok(WorkerEndpoint {
        worker_id: worker_id.to_owned(),
        rank: rank as u32,
        node: host_name()?,
        address_space: format!("{}:{}", std::process::id(), uuid::Uuid::new_v4().simple()),
        incarnation: uuid::Uuid::new_v4().simple().to_string(),
    })
}

/// Bytes of the largest vision feature product the image policy admits: the
/// feature rows of its largest image at the vision encoder's output width.
/// Zero for a model without a vision encoder or an understanding transform.
fn vision_feature_bytes(model: &LoadedModel, model_dtype: &str) -> Result<u64> {
    let declarations = &model.declarations;
    let Some(call) = declarations
        .names()
        .flat_map(|name| declarations.calls(name))
        .find(|call| call.is(Capability::PatchEncoder))
    else {
        return Ok(0);
    };
    let dtype = DType::from_name(model_dtype)
        .ok_or_else(|| Error::Unsupported(format!("unsupported model dtype {model_dtype:?}")))?;
    let EncoderInfo::Vision(vision) = EncoderInfo::describe(model.handle(), &call.path, dtype)?
    else {
        return Ok(0);
    };

    // The encoder merges `downsample` patches on each axis into one feature.
    let rows = u64::from(vision.image_patches / (vision.downsample * vision.downsample));
    Ok(rows * u64::from(vision.output_size) * vision.output_dtype.bytes() as u64)
}

/// Resolve the rank's capacity on the loaded model.
#[allow(clippy::too_many_arguments)]
fn plan(
    launch: &WorkerLaunch,
    config: &mut WorkerConfig,
    model: &LoadedModel,
    image: &ImageModel,
    world: &World,
    budget: Option<&DeviceBudget>,
    components: &Components,
    unit_columns: usize,
    video: Option<&VideoModel>,
) -> Result<(Plan, Option<VideoCapacity>)> {
    let device = config.device;
    if let (Device::Cuda(index), Some(budget)) = (device, budget) {
        config.pool_storage_bytes = Some(
            budget
                .budget(index, config.cache_storage_fraction)?
                .available as usize,
        );
    }

    let declarations = &model.declarations;
    let text = declarations.has(Capability::CausalLm);
    let prefix_cache_dtype = config.prefix_cache_dtype.clone();
    let block_size = page_size(
        config.block_size,
        |size| {
            if text {
                CachePlan::reads_pages(
                    model.handle(),
                    &model.attention_backend,
                    size,
                    prefix_cache_dtype.as_deref(),
                )
            } else {
                Ok(true)
            }
        },
        &world.host,
    )?;
    config.block_size = Some(block_size);

    let dedicated = config.role == "experts";
    let held: Vec<&str> = launch.components.keys().map(String::as_str).collect();
    let supported: Vec<CallKind> = if dedicated {
        Vec::new()
    } else {
        let declared = declarations.supported(&held);
        CallKind::ALL
            .into_iter()
            .filter(|kind| declared.contains(kind) && launch.supported_calls.contains(kind))
            .collect()
    };
    if supported.is_empty() && !dedicated {
        return Err(Error::Unsupported(
            "worker model implements none of the requested call kinds".into(),
        ));
    }
    let media_components = if dedicated {
        BTreeMap::new()
    } else {
        declarations
            .routes(&held)?
            .into_iter()
            .filter(|(kind, _)| supported.contains(&CallKind::Media(*kind)))
            .collect()
    };
    let capacity = match video {
        Some(video) => {
            let sampling = launch.components.contains_key(&video.denoiser);
            let denoising = components.get(&video.denoiser).is_some();
            let decoding = components
                .placed()
                .iter()
                .any(|placed| placed.has(Capability::VideoDecoder));
            Some(crate::capacity::video::plan(
                video, config, sampling, denoising, decoding,
            )?)
        }
        None => None,
    };
    let infos: Vec<ComponentInfo> = launch
        .components
        .iter()
        .map(
            |(name, config): (&String, &ComponentConfig)| ComponentInfo {
                name: name.clone(),
                config: config.clone(),
                // Only a video model's components publish products.
                outputs: capacity
                    .as_ref()
                    .and_then(|capacity| capacity.outputs.get(name).cloned())
                    .unwrap_or_default(),
            },
        )
        .collect();
    let fabric_handles = match device {
        Device::Cuda(index) => HandleType::probe(DeviceId(index))? == HandleType::Fabric,
        Device::Cpu => false,
    };
    let report = Report {
        model_name: model.model_name.clone(),
        endpoint: endpoint(&launch.worker_id, config.rank)?,
        device: config.device.to_string(),
        transfer_backends: launch.data_plane.backends.clone(),
        fabric_handles,
        attention_backend: model.attention_backend.clone(),
        supported_calls: supported,
        media_components,
        components: infos,
        weight_formats: model.weight_formats()?,
        activation_formats: model.activation_formats()?,
    };
    if let (Some(video), Some(capacity)) = (video, capacity) {
        let plan = video_layout(
            report,
            config,
            launch,
            declarations,
            components,
            video,
            &capacity,
            budget,
            &world.host,
        )?;
        return Ok((plan, Some(capacity)));
    }

    let element_bytes = DType::from_name(&config.model_dtype)
        .map(DType::bytes)
        .ok_or_else(|| {
            Error::Unsupported(format!("unsupported latent dtype {:?}", config.model_dtype))
        })?;
    // The longest image sequence an image denoiser accepts, in latent rows.
    let image_tokens = image
        .denoiser
        .map_or(0, |layout| layout.max_sequence_tokens as usize);
    let latent = image.denoiser.map(|layout| {
        LatentPlan::image(
            config.max_request_pool_size,
            image_tokens,
            config.cache_token_capacity,
            block_size,
            layout.width() as usize,
            element_bytes,
        )
    });
    let token_facts = TokenFacts {
        image_processor: image.processing.is_some(),
        vision_feature_bytes: vision_feature_bytes(model, &config.model_dtype)?,
        latent_feature_bytes: latent.map_or(0, |plan| {
            image_tokens as u64 * plan.latent_width as u64 * element_bytes as u64
        }),
        latent,
    };
    let mut layout = TokenLayout::new(report, config, launch.ipc.queue_depth, &token_facts)?;
    // Every rank of a worker holding a causal LM component keeps the same
    // cache pool, including ranks that hold only its other components: the
    // worker advertises token work for all of its ranks, and they agree on
    // one pool capacity.
    let holds_text = held.iter().any(|name| {
        declarations
            .calls(name)
            .iter()
            .any(|call| call.is(Capability::CausalLm))
    });
    let cache = if text && !dedicated && holds_text {
        let cache = CachePlan::plan(model.handle(), block_size, prefix_cache_dtype.as_deref())?;
        let total = match device {
            Device::Cuda(index) => total_bytes(index)?,
            Device::Cpu => 0,
        };
        // The cache's fixed storage beside its units: block tables, import
        // scratch and one unit index column per import worker and lane.
        let shapes = cache.group_shapes()?;
        let width = cache_width(&shapes, config.max_sequence_tokens);
        let import_capacity = layout.info().max_unresolved_calls as usize;
        let fixed = PrefixCache::fixed_bytes(
            &cache,
            0,
            layout.info().request_slots,
            width,
            import_capacity,
            unit_columns,
        )?;
        layout.reserve(config.device, fixed);
        // The text runners' columns and the slot continuation state.
        layout.reserve(
            config.device,
            reserved_bytes(
                model,
                config,
                layout.info().request_slots as usize,
                &cache,
                image.processing.is_some(),
                unit_columns,
            )?,
        );
        // The image denoiser's columns and the image decoder's staging.
        if image.denoiser.is_some() {
            layout.reserve(
                config.device,
                image_reserved_bytes(
                    config,
                    layout.info().request_slots as usize,
                    image.denoiser.as_ref(),
                ),
            );
        }
        // The diffusion LM runners' buffers, banks and sampler workspaces.
        layout.reserve(
            config.device,
            diffusion_lm::reserved_bytes(
                model,
                components,
                expert_exchange(launch),
                config,
                layout.info().request_slots as usize,
                &cache,
            )?,
        );
        let columns = PrefixCache::import_workers(import_capacity) + unit_columns;
        layout.fit_cache(&cache, 8 * columns as u64, config, total, &world.host)?;
        Some(Arc::new(cache))
    } else {
        None
    };
    let mut plan = layout.finish(config, cache);
    if let (Some(cache), Some(info)) = (&plan.cache, &plan.info.prefix_cache) {
        // Batches never exceed the largest captured graph bucket.
        let bounds = TextBounds::new(
            config,
            plan.info.request_slots as usize,
            cache,
            Some(info.num_units),
            device.is_cuda(),
        )?;
        let (prefill, decode) = bounds.graph_calls();
        plan.bound_graph_calls(prefill, decode);
    }
    Ok((plan, None))
}

/// The layout of a video rank: request slots sized by every rank's storage
/// for that many resident requests, agreed over the group.
#[allow(clippy::too_many_arguments)]
fn video_layout(
    report: Report,
    config: &mut WorkerConfig,
    launch: &WorkerLaunch,
    declarations: &crate::config::placement::Declarations,
    components: &Components,
    video: &VideoModel,
    capacity: &VideoCapacity,
    budget: Option<&DeviceBudget>,
    group: &crate::parallel::host::HostGroup,
) -> Result<Plan> {
    let queue_depth = launch.ipc.queue_depth;
    let deployment: Vec<&str> = if config.deployment_components.is_empty() {
        launch.components.keys().map(String::as_str).collect()
    } else {
        config
            .deployment_components
            .iter()
            .map(String::as_str)
            .collect()
    };
    let routes = declarations.routes(&deployment)?;
    let held: std::collections::BTreeSet<String> = components
        .placed()
        .iter()
        .map(|placed| placed.name.clone())
        .collect();
    let local = crate::capacity::video::local_product_bytes(&capacity.outputs, &routes, &held);
    let codec_ranks = launch
        .components
        .get("video_codec")
        .map(|component| component.ranks.len());
    if let (Device::Cuda(index), Some(budget)) = (config.device, budget) {
        let available = budget
            .budget(index, config.cache_storage_fraction)?
            .available;
        let maximum = config.max_request_pool_size;
        let minimum = config.min_request_pool_size.min(maximum).max(1);
        let facts = capacity.facts(video, maximum, local, codec_ranks)?;
        let requirements = crate::capacity::plan::slot_requirements(
            config,
            queue_depth,
            &facts,
            minimum,
            maximum,
        )?;
        let slots = crate::capacity::plan::request_slots(&requirements, minimum, available, group)?;
        config.max_request_pool_size = slots;
        config.max_batch_calls = config.max_batch_calls.min(slots);
        config.max_batch_tokens = config.max_batch_tokens.min(slots);
    }
    let facts = capacity.facts(video, config.max_request_pool_size, local, codec_ranks)?;
    crate::capacity::plan::request_layout(
        report,
        config,
        queue_depth,
        &facts,
        components.holds_host(),
    )
}

/// Whether routed experts use an expert exchange, and the attention ranks of
/// a disaggregated expert union (zero when every rank holds its experts).
fn expert_exchange(launch: &WorkerLaunch) -> (bool, usize) {
    // Weight prefetch moves no tokens.
    let exchanging = launch
        .expert_parallel
        .as_ref()
        .filter(|union| union.exchange != "dwdp");
    exchanging.map_or((false, 0), |union| {
        (
            true,
            if union.disaggregated() {
                union.attention_ranks
            } else {
                0
            },
        )
    })
}

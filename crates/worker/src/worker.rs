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
//! 5. Capacity: block size, request slots, cache units and latent pages, all
//!    agreed over the group; the auxiliary-device check.
//! 6. Request storage and the runners, created by the storage and
//!    capability runtimes on the plan and handed to the backend.
//! 7. The executor and the profiler.
//!
//! [`Worker::warmup`] runs the warmup phases ([`crate::warmup`]) before the
//! rank reports; the engine then warms the serving path with real requests
//! of the shapes the report states. [`Worker::run`] serves the channel, and
//! [`Worker::close`] consumes the rank and releases its owners in dependency
//! order: execution, runners, streams, requests, the process world, then
//! deferred device memory. The launch closes every rank that started,
//! unless the rank abandoned work its threads may wait in for good, which
//! ends the process at once.

mod loading;
mod planning;
mod storage;

pub(crate) use loading::LoadedModel;
pub(crate) use storage::RankStorage;

use std::collections::BTreeMap;

use tvm_ffi::Any;

use uniserve_core::ComponentWeights;
use uniserve_cuda::{DeviceId, PrimaryContext};
use uniserve_ipc::{CallKind, Channel, WorkerEndpoint, WorkerInfo};

use crate::capacity::checks;
use crate::capacity::device::{DeviceBudget, total_bytes};
use crate::capacity::plan::Plan;
use crate::config::placement::{Components, GroupPlacement};
use crate::config::{Device, WorkerConfig, WorkerLaunch};
use crate::cuda_graph::reserve_pools;
use crate::cuda_graph::{GraphStorage, graph_storage_budget_bytes};
use crate::error::WorkerError;
use crate::executor::Executor;
use crate::executor::backend::WorkerBackend;
use crate::executor::transfer::TransferRuntime;
use crate::ffi::Interpreter;
use crate::lane::BLOCK_ON_TIMEOUT;
use crate::parallel::world::{World, startup_timeout};
use crate::parallel::{
    DwdpManager, ExpertStart, ExpertUnion, ExpertWarmup, HostWeights, PeerGroup,
};
use crate::profiling::kernels::{KernelRecords, KernelWatch};
use crate::profiling::profiler::{ProfileConfig, Profiler};
use crate::profiling::range;
use crate::request::video::VideoRequests;
use crate::runner::causal_lm::CausalLMRunners;
use crate::runner::causal_lm::start::{Startup, expert_tokens};
use crate::runner::causal_lm::warmup::CausalLMWarmup;
use crate::runner::component::VideoSetup;
use crate::runner::decoder::warmup::DecoderWarmup;
use crate::runner::denoiser::image_model::ImageModel;
use crate::runner::denoiser::warmup::DenoiserWarmup;
use crate::runner::denoiser::{VideoModel, placed_denoiser};
use crate::runner::diffusion_lm::{self, DiffusionLMRunner, DiffusionLMWarmup};
use crate::runner::encoder::warmup::EncoderWarmup;
use crate::runner::scratch::Scratches;
use crate::runner::start::{ImageOwners, Startup as ImageStartup, start_video};
use crate::service::{Ending, Service};
use crate::stream::{CUDAStream, Streams};
use crate::transfer::TransportKind;
use crate::warmup::WarmupHooks;
use crate::{Error, Result};
use loading::{ComponentMesh, Description, ExpertPartition, head_projection, sequence_kind};

use planning::{cache_width, log_capacity, plan};
use storage::allocate_storage;

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
    /// The shared work areas of every stream runners construct their
    /// execution contexts on.
    _scratches: Scratches,
    executor: Executor,
    hooks: Vec<Box<dyn WarmupHooks>>,
    /// Retained for the rank's lifetime: torch, the driver owners and the
    /// reclaimer all use the device's primary context.
    context: Option<PrimaryContext>,
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
        let model = LoadedModel::load(description, &launch, meshes, experts, host_only)
            .map_err(phase("load"))?;
        // Every later graph decision of the rank reads the resolved policy.
        let host_component = components
            .placed()
            .iter()
            .find(|placed| placed.config.weights == ComponentWeights::Host)
            .map(|placed| placed.name.as_str());
        launch.execution.graph_policy =
            graph_policy(&launch.execution.graph_policy, &model, host_component)
                .map_err(phase("load"))?;
        let mut config = launch.execution.clone();
        if model.video_decoder {
            config = config
                .for_media_slots(launch.ipc.queue_depth)
                .map_err(phase("load"))?;
        }
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
        // Host-resident layer weights release their device storage before
        // capacity planning measures free memory.
        let host_weights = HostWeights::start(&model, &components).map_err(phase("load"))?;
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

        let storage =
            allocate_storage(&launch, &config, &plan, &streams).map_err(phase("storage"))?;
        let mut components = components;
        let preparation = device.cuda().and_then(|cuda| streams.preparation(cuda));
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
                        queue_depth: launch.ipc.queue_depth,
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
                let leases = 2 * launch.ipc.queue_depth + 2;
                let words = info.max_batch_calls.max(1) as usize;
                Some((
                    VideoRequests::new(video, capacity.media).map_err(phase("storage"))?,
                    encoders,
                    denoisers,
                    decoders,
                    media,
                    crate::OutputPool::new(device, leases, words).map_err(phase("storage"))?,
                ))
            }
            _ => None,
        };
        // The shapes the engine warms the serving path with, through its
        // scheduler, once the rank reported, and the image-unconditional
        // guidance prefix it reserves.
        let info = {
            let mut info = info;
            if let Some(image) = &image {
                info.image_warmups = image
                    .warmups(info.request_slots, info.latent_capacity_units())
                    .map_err(phase("bind"))?;
                info.image_unconditional_token_ids =
                    image.image_unconditional_prefix().map_err(phase("bind"))?;
            }
            info.diffusion_lm_step_rows = diffusion_lm
                .iter()
                .map(DiffusionLMRunner::step_rows)
                .max()
                .unwrap_or(0);
            info
        };
        let placement = GroupPlacement {
            components: launch.components.clone(),
            world_size: config.world_size,
        };
        let mut backend = WorkerBackend::new(info.clone(), config.rank, placement, components)
            .map_err(phase("storage"))?;
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
        backend.set_host_weights(host_weights);
        let expert_warmup = experts.is_some();
        if let Some(experts) = experts {
            backend.set_experts(experts);
        }
        let cuda: Vec<DeviceId> = devices.iter().filter_map(|device| device.cuda()).collect();
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
        let mut hooks: Vec<Box<dyn WarmupHooks>> = Vec::new();
        if video_warmup || image_warmup {
            // Prepared and captured in this order on every rank.
            hooks.push(Box::new(EncoderWarmup));
            hooks.push(Box::new(DenoiserWarmup));
            hooks.push(Box::new(DecoderWarmup));
        }
        if text_warmup {
            hooks.push(Box::new(CausalLMWarmup {
                row_tokens: config.max_model_len.min(config.max_batch_tokens).max(1),
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
            _scratches: scratches,
            executor,
            hooks,
            context,
            warmed: false,
        })
    }

    /// Run warmup once, before the first serving submission.
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
        for hooks in &mut self.hooks {
            hooks.prepare(executor.backend_mut())?;
        }
        {
            let _capture = range(c"uniserve.worker.capture", None);
            for hooks in &mut self.hooks {
                hooks.capture(executor.backend_mut())?;
            }
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

    /// Serve the channel until the engine closes the rank or a fatal
    /// response ends admission, or until the head's channel fails
    /// ([`Ending`]), reporting a lane that stays inside one job for
    /// `stall_bound`.
    pub(crate) fn run(
        &mut self,
        channel: &mut Channel,
        stall_bound: std::time::Duration,
    ) -> std::result::Result<Ending, WorkerError> {
        self.warmup()?;
        let info = self.info.clone();
        let executor = &mut self.executor;
        // Ranks of an expert union join every step outside their own.
        let experts = executor.backend().experts().is_some();
        executor.backend().set_storage_wake(Some(channel.wake()));
        let served = Service::new(executor, channel, info, experts, stall_bound).run();
        executor.backend().set_storage_wake(None);
        served
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
            // The lanes first perform what is still queued on them (work
            // nobody waited for, like a lifecycle command's), so that no
            // job addresses an owner while it closes.
            let drained = self.executor.backend().job_queues().drain(BLOCK_ON_TIMEOUT);
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
    graphs: &mut GraphStorage,
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
        let reserved = reserve_pools(graphs, &create, stream, &[device], None)?;
        pools.extend(reserved.first().map(|pool| pool.value.clone()));
    }
    experts.set_pools(pools);
    Ok(())
}

/// The rank's identity for transfers: host, process and incarnation. The
/// engine knows which worker and rank the incarnation serves.
fn endpoint() -> Result<WorkerEndpoint> {
    Ok(WorkerEndpoint {
        node: host_name()?,
        address_space: format!("{}:{}", std::process::id(), uuid::Uuid::new_v4().simple()),
        incarnation: uuid::Uuid::new_v4().simple().to_string(),
    })
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

/// Whether the rank publishes device products over CUDA VMM.
fn exports_device_products(launch: &WorkerLaunch) -> bool {
    launch
        .data_plane
        .exports
        .iter()
        .any(|name| name == TransportKind::CudaVmm.name())
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

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
//! 5. Capacity: page size, request slots, KV units and latent pages, all
//!    agreed over the group; the auxiliary-device check.
//! 6. Request storage and component bindings, created by the storage and
//!    capability runtimes on the plan and registered with the backend.
//! 7. The executor and the profiler.
//!
//! [`Worker::warmup`] runs the warmup phases ([`crate::warmup`]),
//! [`Worker::run`] serves the channel, and [`Worker::close`] releases owners
//! in dependency order: execution, bindings, streams, requests, the process
//! world, then deferred device memory.

mod lanes;
mod storage;

pub use lanes::{Lane, Lanes};
pub use storage::RankStorage;

use std::collections::BTreeMap;

use std::sync::Arc;
use tvm_ffi::Any;
use uniserve_core::ComponentConfig;

use uniserve_cuda::{DeviceId, HandleType, PrimaryContext, Stream};
use uniserve_ipc::{CallKind, ComponentInfo, ForwardMode, WorkerEndpoint, WorkerInfo};

use crate::capacity::checks;
use crate::capacity::device::{DeviceBudget, total_bytes};
use crate::capacity::plan::{
    LatentPlan, Plan, Report, TokenFacts, TokenLayout, page_size, product_bytes,
};
use crate::config::{Device, WorkerConfig, WorkerLaunch};
use crate::error::WorkerError;
use crate::executor::Executor;
use crate::executor::backend::{KernelWatch, WorkerBackend, binding_label};
use crate::ffi::{DType, Handle, Record};
use crate::launch::Endpoint;
use crate::model::components::Capability;
use crate::model::interface::Interface;
use crate::model::registry::Components;
use crate::model::{
    ComponentMesh, Description, ExpertPartition, LoadedModel, ModelFacts, sequence_kind,
};
use crate::parallel::world::{World, startup_timeout};
use crate::profiling::kernels::KernelRecords;
use crate::profiling::profiler::{ProfileConfig, Profiler};
use crate::profiling::range;
use crate::runner::{GraphStorage, graph_storage_budget_bytes};
use crate::service::Service;
use crate::storage::{GroupShape, KVCache, KVPlan};
use crate::stream::{CUDAStream, EventPool};
use crate::transfer::{Capacity, TransportKind, TransportOptions, Transports};
use crate::warmup::{Scenarios, WarmupHooks};
use crate::{Error, Result};

/// One caching-allocator segment: device, graph pool id and bytes.
type Segment = (i32, (u64, u64), u64);

/// A startup failure, with the phase that failed.
fn phase(phase: &'static str) -> impl Fn(Error) -> WorkerError {
    move |error| {
        // Description and process-group setup refuse a launch the rank
        // cannot serve; report input errors there as unsupported setups.
        let error = match error {
            Error::Numerical {
                kind,
                message,
                traceback,
                function,
            } if matches!(phase, "describe" | "process groups")
                && matches!(
                    kind.as_str(),
                    "InputError" | "FileNotFound" | "ValueError" | "FileNotFoundError"
                ) =>
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
    launch: WorkerLaunch,
    interface: Interface,
    config: WorkerConfig,
    info: WorkerInfo,
    plan: Plan,
    model: Option<LoadedModel>,
    world: Option<World>,
    budget: Option<DeviceBudget>,
    lanes: Lanes,
    graphs: GraphStorage<String, Handle>,
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
    pub fn start(
        launch: WorkerLaunch,
        interface: Interface,
    ) -> std::result::Result<Self, WorkerError> {
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
        let description = Description::describe(&interface, &launch).map_err(phase("describe"))?;
        let resolved = description
            .resolve(&interface, &launch)
            .map_err(phase("describe"))?;

        drop(describing);
        let mut world = World::initialize(&interface, &launch, startup_timeout())
            .map_err(phase("process groups"))?;
        let dedicated = launch.execution.role == "experts";
        if !dedicated {
            world
                .bind_components(&interface, &launch, &description.declarations)
                .map_err(phase("process groups"))?;
        }

        let mut config = launch.execution.clone();
        let components = Components::new(
            &description.declarations,
            &launch.components,
            config.rank,
            device,
            config
                .generation_device
                .as_deref()
                .and_then(|name| Device::parse(name, "device").ok()),
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
        let model = LoadedModel::load(
            &interface,
            description,
            resolved,
            &launch,
            meshes,
            experts,
            host_only,
        )
        .map_err(phase("load"))?;
        let facts = model.facts(&interface, &launch).map_err(phase("load"))?;
        drop(loading);
        if launch.stub {
            // The stub encodes many small images in tests.
            config.encoder_cache_entries = 1024;
        }

        let devices: Vec<Device> = config
            .devices()
            .filter_map(|name| Device::parse(name, "device").ok())
            .collect();
        let lanes = Lanes::create(&config, &devices, launch.ipc.queue_depth + 1)
            .map_err(phase("streams"))?;
        let budget = match device {
            Device::Cuda(_) => Some(
                DeviceBudget::new(interface.release_cached.clone()).map_err(phase("capacity"))?,
            ),
            Device::Cpu => None,
        };

        let planning = range(c"uniserve.worker.plan", None);
        let plan = plan(
            &launch,
            &interface,
            &mut config,
            &model,
            &facts,
            &world,
            budget.as_ref(),
            &components,
            lanes.lanes().len().max(1),
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

        let storage = allocate_storage(&launch, &interface, &config, &plan, &lanes)
            .map_err(phase("storage"))?;
        let mut backend =
            WorkerBackend::new(info.clone(), config.rank, components).map_err(phase("storage"))?;
        backend.add_storage(Box::new(storage));
        let profiler = ProfileConfig::from_env(|name| std::env::var(name).ok())
            .map_err(phase("profiling"))?
            .map(|profile| {
                Profiler::new(
                    profile,
                    interface.profiler_start.clone(),
                    interface.profiler_stop.clone(),
                    device.cuda_index(),
                )
            });
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
        tracing::info!(target: "uniserve_worker.launch", "loaded numerical model {}", facts.model_name);
        Ok(Self {
            launch,
            interface,
            config,
            info,
            plan,
            model: Some(model),
            world: Some(world),
            budget,
            lanes,
            graphs: GraphStorage::new(budgets),
            executor: Some(executor),
            hooks: Vec::new(),
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

    pub fn interface(&self) -> &Interface {
        &self.interface
    }

    pub fn model(&self) -> Option<&LoadedModel> {
        self.model.as_ref()
    }

    pub fn world(&self) -> Option<&World> {
        self.world.as_ref()
    }

    pub fn lanes_mut(&mut self) -> &mut Lanes {
        &mut self.lanes
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
        let cuda = self.config.primary_device().is_cuda();

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
        // Serving runs Python only inside interface calls; collection pauses
        // there would stall steps, so the heap built so far is frozen.
        self.interface
            .freeze_gc
            .invoke(&[])
            .map_err(phase("warmup"))?;
        self.warmed = true;
        Ok(())
    }

    /// Serve the channel until the engine closes the rank or a fatal
    /// response ends admission.
    pub fn run(&mut self, endpoint: &mut Endpoint) -> std::result::Result<(), WorkerError> {
        self.warmup()?;
        let experts = self.launch.expert_parallel.is_some();
        let info = self.info.clone();
        let executor = self
            .executor
            .as_mut()
            .ok_or_else(|| WorkerError::from(Error::State("the worker is closed")))?;
        executor
            .backend()
            .set_storage_wake(Some(endpoint.completion_wake()));
        let served = Service::new(executor, endpoint.server(), info, experts).run();
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
            for stream in self.lanes.streams() {
                record(stream.synchronize().map_err(WorkerError::from));
            }
        }
        if let Some(mut executor) = self.executor.take() {
            executor.backend_mut().close();
            if !aborted {
                record(executor.close());
            }
            let closed = executor.backend_mut().components_mut().close(aborted);
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
        record(self.lanes.close(aborted).map_err(WorkerError::from));
        if let Some(mut world) = self.world.take() {
            record(world.close(aborted).map_err(WorkerError::from));
        }
        if aborted {
            #[allow(clippy::mem_forget)]
            std::mem::forget(self.model.take());
            #[allow(clippy::mem_forget)]
            std::mem::forget(self.context.take());
        } else {
            self.model = None;
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
        let devices: Vec<String> = self.config.devices().map(str::to_owned).collect();
        let products = self.plan.arena.device_product_bytes / devices.len().max(1) as u64;
        for name in devices {
            let Ok(Device::Cuda(index)) = Device::parse(&name, "device") else {
                continue;
            };
            let measured = budget.budget(index, self.config.kv_storage_fraction)?;
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

    /// Allocator segments of `device`.
    fn segments(&self, device: i32) -> Result<Vec<Segment>> {
        let stats = Record::new(
            "memory.stats",
            self.interface
                .memory_stats
                .call_values(&[Any::from(i64::from(device))])?,
        )?;
        stats
            .list("segments")?
            .into_iter()
            .map(|segment| {
                let values = crate::ffi::integers(segment)
                    .map_err(|error| Error::numerical("memory.stats", &error))?;
                let [first, second, bytes] = values[..] else {
                    return Err(Error::Interface(
                        "memory.stats segments are [pool, pool, bytes]".into(),
                    ));
                };
                Ok((device, (first as u64, second as u64), bytes as u64))
            })
            .collect()
    }

    fn require_codecs(&self) -> std::result::Result<(), WorkerError> {
        let function = self
            .interface
            .optional("media.require_codecs")
            .ok_or_else(|| {
                WorkerError::from(Error::Unsupported(
                    "video encoding and muxing are served by the host media runtime".into(),
                ))
            })?;
        function.invoke(&[]).map_err(phase("warmup"))
    }

    /// Check graph residency, report memory and kernels, then seal.
    fn complete_startup(&mut self) -> std::result::Result<(), WorkerError> {
        let cuda_devices: Vec<i32> = self
            .config
            .devices()
            .filter_map(|name| {
                Device::parse(name, "device")
                    .ok()
                    .and_then(Device::cuda_index)
            })
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
                let stats = Record::new(
                    "memory.stats",
                    self.interface
                        .memory_stats
                        .call_values(&[Any::from(i64::from(device))])
                        .map_err(phase("warmup"))?,
                )
                .map_err(phase("warmup"))?;
                let reserved = stats.count("reserved").map_err(phase("warmup"))?;
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
        let mut records = KernelRecords::default();
        for binding in executor.backend().components().bindings() {
            let label = binding_label(binding);
            let kernels = binding.kernels().map_err(phase("warmup"))?;
            records
                .add(&label, &binding.device.to_string(), &kernels)
                .map_err(phase("warmup"))?;
        }
        records.log("startup", self.config.rank, &self.config.device);
        let choices: i64 = self
            .interface
            .kernel_choices
            .call_as(&[])
            .map_err(phase("warmup"))?;
        executor.backend_mut().watch_kernels(KernelWatch {
            choices: self.interface.kernel_choices.clone(),
            last: choices,
            records,
            rank: self.config.rank,
            device: self.config.device.clone(),
        });
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
/// transports and the KV cache of a text model, whose planes are reset on
/// the device's preparation stream.
fn allocate_storage(
    launch: &WorkerLaunch,
    interface: &Interface,
    config: &WorkerConfig,
    plan: &Plan,
    lanes: &Lanes,
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
    let transports = Transports::new(&kinds, capacity, Arc::clone(&events), &options)?;

    let device = config.primary_device().cuda_index();
    let preparation = device.and_then(|index| lanes.preparation(DeviceId(index)));
    let kv = match (&plan.cache, &plan.info.kv_cache) {
        (Some(cache), Some(info)) => Some(KVCache::new(
            Arc::clone(cache),
            info.num_units,
            plan.info.request_slots,
            cache_width(&cache.group_shapes()?, config.max_sequence_tokens),
            plan.info.max_unresolved_calls as usize,
            device,
            preparation.map(CUDAStream::stream).transpose()?,
            &interface.function("cache.bind_units")?,
            &interface.function("cache.bind_conversion")?,
        )?),
        _ => None,
    };
    if let Some(stream) = preparation {
        stream.synchronize()?;
    }
    // Slot releases order after the token batches that read them.
    let release_stream = match device {
        Some(index) => lanes
            .stream_for(CallKind::Forward(ForwardMode::Decode), DeviceId(index))
            .map(|stream| -> Result<Stream> {
                // SAFETY: the lanes own the stream and close after the
                // storage owners (see `Worker::close`).
                Ok(unsafe { Stream::borrowed(DeviceId(index), stream.handle()?) }?)
            })
            .transpose()?,
        None => None,
    };
    Ok(RankStorage::new(events, transports, kv, release_stream))
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
    let kv = info.kv_cache.as_ref().map_or_else(
        || "no KV pool".to_owned(),
        |cache| {
            format!(
                "{} KV units of {} bytes in {} groups",
                cache.num_units,
                cache.unit_bytes,
                cache.groups.len()
            )
        },
    );
    tracing::info!(
        target: "uniserve_worker.launch",
        "planned capacity: {kv}, KV page size {} tokens, {} request slots, {} calls and {} tokens per batch, {} latent pages, {} buffer pool bytes",
        config.block_size.unwrap_or(0),
        info.request_slots,
        info.max_batch_calls,
        info.max_batch_tokens,
        info.latent_pages,
        info.buffer_pool_bytes,
    );
}

/// The rank's identity for transfers: host, process and incarnation.
fn endpoint(worker_id: &str, rank: usize) -> WorkerEndpoint {
    let mut name = [0u8; 256];
    // SAFETY: gethostname writes at most `name.len()` bytes.
    let node = if unsafe { libc::gethostname(name.as_mut_ptr().cast(), name.len()) } == 0 {
        let end = name
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(name.len());
        String::from_utf8_lossy(&name[..end]).into_owned()
    } else {
        "localhost".to_owned()
    };
    WorkerEndpoint {
        worker_id: worker_id.to_owned(),
        rank: rank as u32,
        node,
        address_space: format!("{}:{}", std::process::id(), uuid::Uuid::new_v4().simple()),
        incarnation: uuid::Uuid::new_v4().simple().to_string(),
    }
}

/// Resolve the rank's capacity on the loaded model.
#[allow(clippy::too_many_arguments)]
fn plan(
    launch: &WorkerLaunch,
    interface: &Interface,
    config: &mut WorkerConfig,
    model: &LoadedModel,
    facts: &ModelFacts,
    world: &World,
    budget: Option<&DeviceBudget>,
    components: &Components,
    unit_columns: usize,
) -> Result<Plan> {
    let device = config.primary_device();
    if let (Device::Cuda(index), Some(budget)) = (device, budget) {
        config.pool_storage_bytes =
            Some(budget.budget(index, config.kv_storage_fraction)?.available as usize);
    }

    let declarations = &model.declarations;
    let text = declarations.has(Capability::CausalLm);
    let kv_dtype = config.kv_cache_dtype.clone();
    let block_size = page_size(
        config.block_size,
        |size| {
            if text {
                let plan = model.cache_plan(interface, size, kv_dtype.as_deref())?;
                model.reads_pages(interface, &facts.attention_backend, &plan)
            } else {
                Ok(true)
            }
        },
        &world.host,
    )?;
    config.block_size = Some(block_size);
    if model.video_decoder || declarations.has(Capability::VideoPostprocessor) {
        return Err(Error::Unsupported(
            "video models are planned and served by the video runtime".into(),
        ));
    }

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
    let infos: Vec<ComponentInfo> = launch
        .components
        .iter()
        .map(
            |(name, config): (&String, &ComponentConfig)| ComponentInfo {
                name: name.clone(),
                config: config.clone(),
                outputs: facts.outputs.get(name).cloned().unwrap_or_default(),
            },
        )
        .collect();
    let fabric_handles = match device {
        Device::Cuda(index) => HandleType::probe(DeviceId(index))? == HandleType::Fabric,
        Device::Cpu => false,
    };
    let report = Report {
        model_name: facts.model_name.clone(),
        endpoint: endpoint(&launch.worker_id, config.rank),
        device: config.device.clone(),
        transfer_backends: launch.data_plane.backends.clone(),
        fabric_handles,
        attention_backend: facts.attention_backend.clone(),
        supported_calls: supported,
        media_components,
        components: infos,
        weight_formats: facts.weight_formats.clone(),
        activation_formats: facts.activation_formats.clone(),
    };

    let element_bytes = DType::from_name(&config.model_dtype)
        .map(DType::bytes)
        .ok_or_else(|| {
            Error::Unsupported(format!("unsupported latent dtype {:?}", config.model_dtype))
        })?;
    let latent = facts.image_flow.map(|flow| {
        LatentPlan::image(
            config.max_request_pool_size,
            flow.max_tokens,
            config.kv_token_capacity,
            block_size,
            flow.latent_channels * flow.patch_size * flow.patch_size,
            element_bytes,
        )
    });
    let token_facts = TokenFacts {
        image_processor: facts.image_processor,
        vision_feature_bytes: facts.vision_feature_bytes,
        latent_feature_bytes: latent.map_or(0, |plan| {
            facts.image_flow.map_or(0, |flow| flow.max_tokens) as u64
                * plan.latent_width as u64
                * element_bytes as u64
        }),
        latent,
        product_bytes: facts
            .outputs
            .values()
            .map(|outputs| product_bytes(outputs))
            .sum(),
    };
    let mut layout = TokenLayout::new(report, config, launch.ipc.queue_depth, &token_facts)?;
    let cache = if text
        && !dedicated
        && components
            .placed()
            .iter()
            .any(|placed| placed.has(Capability::CausalLm))
    {
        let cache =
            KVPlan::decode(model.cache_plan(interface, block_size, kv_dtype.as_deref())?)?;
        let total = match device {
            Device::Cuda(index) => total_bytes(index)?,
            Device::Cpu => 0,
        };
        // The cache's fixed storage beside its units: block tables, import
        // scratch and one unit index column per import worker and lane.
        let shapes = cache.group_shapes()?;
        let width = cache_width(&shapes, config.max_sequence_tokens);
        let import_capacity = layout.info().max_unresolved_calls as usize;
        let fixed = KVCache::fixed_bytes(
            &cache,
            0,
            layout.info().request_slots,
            width,
            import_capacity,
            unit_columns,
        )?;
        layout.reserve(&config.device, fixed);
        let columns = KVCache::import_workers(import_capacity) + unit_columns;
        layout.fit_cache(&cache, 8 * columns as u64, config, total, &world.host)?;
        Some(Arc::new(cache))
    } else {
        None
    };
    Ok(layout.finish(config, cache))
}

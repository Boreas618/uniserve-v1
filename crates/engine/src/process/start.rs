//! Starting and closing a main process: the control thread's connections, the ranks' load, the
//! device workers, the schedulers, and a standby's park; and the orderly close that ends them.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use ops::Generation;
use program_server::Client;
use program_server::program::Programs;
use proto::config::EngineArgs;
use worker::blocking::{BlockingKind, ThreadPool};
use worker::control::{Control, Endpoints};
use worker::device::Worker;
use worker::hot_update::HotUpdate;
use worker::runners::products::{GroupProducts, RankProducts};
use worker::watchdog::Watchdog;

use super::MainProcess;
use super::claim::claim;
use super::device_command;
use super::launch::{Epoch, ProcessInterface, handshake};
use super::ranks::{LoadedRank, load_ranks, schedulers};
use super::reports::{announce, state_failure};
use super::serve::Wait;
use crate::programs::modules::ModuleWork;
use crate::rank::ModelRunner;

impl MainProcess {
    /// Connects to the daemon, loads the epoch's model, and starts the epoch's threads in the order
    /// the architecture lists them (`docs/architecture.md`, The threads of a main process).
    ///
    /// The epoch's configuration is resolved against this process's launch first
    /// ([`ModelRunner`]), because three of the things the main process needs are the
    /// node's rather than the epoch's: which devices it serves, where its Program
    /// Server is, and what those devices measure.
    pub(super) fn start(launch: &EngineArgs, catalogue: &crate::Catalogue) -> Result<Self> {
        let build = &launch.engine_config;
        let family = catalogue.resolve(Path::new(&launch.engine_config.model.model))?;
        let model_runner = ModelRunner::of(launch, family)?;
        let mut control = Control::connect(
            &Endpoints {
                requests: launch.requests.clone(),
                peers: launch.peers.clone(),
            },
            launch.engine,
        )?;
        // The request interface's handshake: this process says who it is, and the
        // daemon answers with the epoch this node serves.
        // The epoch this rank belongs to is the daemon's, and both of its facts decide the storage
        // this process claims and the peers it exchanges products with. A zero epoch names no
        // epoch, and a peer list on a world of one is a second node this epoch does not have: each
        // is refused here, by name, rather than at the claim or the first transfer.
        if launch.epoch == 0 {
            bail!("the launch file states epoch 0, which names no epoch for this rank to serve");
        }
        if launch.world_size == 1 && !launch.peers.is_empty() {
            bail!(
                "the launch file states a world of one rank and {} peer address(es): a peer is \
                 another main process of the same world, and this epoch has no other",
                launch.peers.len()
            );
        }
        let epoch = handshake(&mut control, launch)?;
        // The daemon's launch path blocks until this process states what it is, and it
        // sends the epoch's first program-layer command only once it has. `Starting` goes out
        // before the model load, which is minutes long, so the daemon's log and the caller's show a
        // process that is alive rather than one that vanished after the storage claim; the
        // `Ready`/`Standby` at the end of `start` is what lets the epoch begin.
        announce(
            &mut control,
            proto::request::wire::EngineState::Starting,
            None,
        )?;
        // The daemon waits for what this process became, so a start that fails says why: the
        // process ends and its connection closes, which says only that it went.
        let mut control = Some(control);
        let started = Self::load(&mut control, model_runner, epoch, launch, build);
        if let (Err(error), Some(control)) = (&started, control.as_mut()) {
            state_failure(control, &format!("{error:#}"));
        }
        started
    }

    /// Everything of a start past the handshake. It takes `control` for the process it builds, and
    /// leaves it to the caller when it fails before that.
    fn load(
        control: &mut Option<Control>,
        model_runner: ModelRunner,
        epoch: Epoch,
        launch: &EngineArgs,
        build: &proto::config::EngineConfig,
    ) -> Result<Self> {
        let connected = control
            .as_mut()
            .context("a main process starts on the connection it opened")?;
        // The daemon's two interfaces are two sockets: the epoch's requests and metrics travel on
        // the request connection above, and everything the daemon asks of this process - the claim,
        // the program-layer reports, and every command of the program layer - travels on this one.
        let mut process = ProcessInterface::connect(
            &launch.processes,
            &launch.engine_config.deployment,
            launch.engine,
        )?;
        // Every command of the program layer arrives on this socket, and a stage waits for the
        // command before it: the control thread sleeps on it as it does on the request socket.
        connected.watch_process(Some(process.connection.descriptor()));

        // The rank's Storage Server and its claim are the node's, which the daemon answers for;
        // the retained weights outlive the epoch, so the store is built once, before the model
        // loads, and keeps what it maps for the process's life.
        // A host tier is the build config's, and it needs prefix caching to have pages to keep.
        let host_kv_cache = build.kv_cache.host_tier && build.kv_cache.prefix_caching;
        // CUPTI accepts one subscriber per process, and what it observes begins at the subscription,
        // so the process subscribes before it loads a module (`docs/profiling.md`).
        worker::profiling::counters::subscribe();
        // This process serves every rank of its node, each on its own device.
        let rank_runners = model_runner.ranks()?;
        let rank_numbers: Vec<u32> = rank_runners.iter().map(ModelRunner::rank).collect();
        let storage = claim(
            &mut process,
            launch,
            &epoch.host,
            host_kv_cache,
            &rank_numbers,
        )?;
        let ranks = load_ranks(rank_runners, &storage, launch, &epoch)?;

        // The resident owners are opened before any thread that borrows them, and
        // released in reverse order by `close`. The weights are mapped before the capacity plan
        // measures, so the plan sees a device with the model already in it - the figure it computes
        // is what is left, not what was there before the load. Pools sized first and weights after
        // would over-commit the device by exactly the model's size.
        let blocking = ThreadPool::new(model_runner.blocking_limits());
        let modules = blocking.lane::<ModuleWork>(BlockingKind::ModuleLoad)?;
        let programs = Arc::new(Mutex::new(Client::connect(model_runner.program_server())?));

        // The watchdog exists before any device command: `start`'s own commands
        // are bounded by the same stall bound as every command the control thread
        // issues afterwards.
        let watchdog = Watchdog::new(
            launch
                .watchdog_timeout()
                .unwrap_or_else(|| model_runner.watchdog_timeout()),
        )?;
        let update = HotUpdate::new(epoch.generation);

        let mut workers = Vec::with_capacity(model_runner.devices().len());
        let mut generations = Vec::with_capacity(model_runner.devices().len());
        // A rank with no device serves nothing, and saying it here names the deployment's fault
        // rather than leaving the pool's figure to a plan that never sized one.
        if model_runner.devices().is_empty() {
            bail!("this rank serves no device");
        }
        // Every capture of this process carries the probes the deployment names, and the graphs
        // keep them for good: only their switch changes while the process serves.
        if !epoch.profiling.fence_components.is_empty() {
            worker::profiling::fences::Fences::install(
                model_runner.devices(),
                epoch.profiling.fence_components.clone(),
                epoch.profiling.fence_probes,
                &epoch.profiling.fence_probes_off,
            )
            .context("installing the fence probes")?;
        }
        // CUPTI attributes a graph kernel's PC samples only when sampling was enabled before the
        // graph was captured.
        if epoch.profiling.graph_pc_sampling {
            for device in model_runner.devices() {
                let context = platform::PrimaryContext::retain(*device)?;
                worker::profiling::pc_sampling::enable(&context)
                    .context("enabling PC sampling for this rank's graphs")?;
            }
        }
        let mut memories = Vec::with_capacity(ranks.len());
        let mut clients = Vec::with_capacity(ranks.len());
        let mut models = Vec::with_capacity(ranks.len());
        let mut inventories = Vec::with_capacity(ranks.len());
        let mut data_ranks = Vec::with_capacity(ranks.len());
        let first_plan = ranks
            .first()
            .map(|rank| rank.model_runner.scheduled())
            .context("this process serves no rank")?;
        // Each rank of an expert union is a scheduler's own; the ranks of a media group are one
        // pool, which the first rank's plan describes.
        let union = first_plan.union().map(|(union, _)| Arc::clone(union));
        let mut scheduled = Vec::with_capacity(ranks.len());
        // The group's products, which every rank's runners reach through their own view of them.
        let group_products = GroupProducts::new(ranks.len());
        let rank_count = ranks.len() as u32;
        for (rank_index, rank) in ranks.into_iter().enumerate() {
            let LoadedRank {
                model_runner: rank_runner,
                weights,
                client,
                model,
                mut memory,
                mut inventory,
                graph,
                groups,
            } = rank;
            // The scheduler sees the group as one pool, and each rank encodes its own units, so the
            // pool holds every rank's host tasks.
            inventory.host_task_capacity *= rank_count;
            // The device's one pool set is the resident owners' allocator: it is opened with
            // the memory and handed to the worker thread, so every owner and every runner books
            // against the same allowance.
            let pools = memory.pools();
            // The scratch pool's figure comes from the one call that sized it, with the memory: the
            // shadow region and the state group's snapshot are one reservation, and a second
            // computation here could disagree with the pool the allocator reserved.
            let scratch = graph.scratch;
            // The shadow region alone: the twins allocate it, and the reservation beside it holds
            // the state group's snapshot too.
            let shadow = graph.shadow;
            let products: Arc<dyn worker::runners::products::Products> =
                Arc::new(RankProducts::new(Arc::clone(&group_products), rank_index));
            // What the rank's trajectories are computed with, which keys the pages of its host latent
            // pool: the weights it loaded.
            let identity = worker::data_plane::identity_of(&weights);
            for device in rank_runner.devices() {
                let generation = Arc::new(Generation::new(epoch.generation, std::iter::empty()));
                // The scheduler counts this process's ranks as one, the first rank's report, so the
                // first rank states the host latent pool for all of them.
                let runners = rank_runner.runners(
                    *device,
                    &model,
                    &mut memory,
                    &generation,
                    &blocking,
                    &groups,
                    &client,
                    &identity,
                    (rank_index == 0).then_some(&mut inventory),
                )?;
                let worker = Worker::spawn(
                    *device,
                    runners,
                    rank_runner.events(),
                    Arc::clone(&pools),
                    Arc::clone(&products),
                    scratch,
                    shadow,
                )?;
                // The precopy shadows a replaced operator beside the serving one, so the region it
                // runs in is mapped before any batch: the worker books it from the scratch pool
                // the graph storage sized. Nothing is serving yet, so the control thread waits here;
                // `serve` never does.
                //
                // The warmup is not here: the epoch's first generation holds no kernel until the
                // daemon's precopy fills it - and a precopy that names no kernel (the startup
                // switch) fills it from the store's built programs - so there is no bucket to warm
                // yet. It runs at the first switchover, which is the boundary the first request is
                // served at.
                let shadow = worker.set_shadow()?;
                device_command(&watchdog, *device, "shadow region", shadow)
                    .with_context(|| format!("mapping the shadow region of device {}", device.0))?;
                // The park's and the wake's commands reach the Storage Server's ledger from the
                // device thread, which owns the stream a wake resets its units on, so the client is
                // bound beside the cache handle rather than kept only here.
                worker.bind_storage(Arc::clone(&client));
                if let Some((union, member)) = rank_runner.union() {
                    union.bind(
                        member,
                        Arc::clone(worker.queue()),
                        worker::runner::RunnerId(0),
                    )?;
                }
                let worker = Arc::new(worker);
                data_ranks.push(worker::data_plane::Rank {
                    rank: rank_runner.rank(),
                    worker: Arc::downgrade(&worker),
                    device: *device,
                    storage: Arc::clone(&client),
                    cache: memory.shared_cache(),
                    model: Arc::new(crate::rank::RankModel::new(rank_runner.clone())),
                    serving: weights.clone(),
                    host: epoch.host.clone(),
                });
                workers.push(worker);
                generations.push(generation);
            }
            scheduled.push(rank_runner.scheduled());
            memories.push(memory);
            clients.push(client);
            models.push(model);
            inventories.push(inventory);
        }

        // The quota comes after the load: what a process holds to serve at all is the deployment's
        // statement, and what the quota governs is what it retains beside that.
        for rank in &data_ranks {
            rank.storage
                .state_quota(rank.device, build.mem_fraction_static)
                .context("stating this rank's quota of its device")?;
        }
        // The storage operations and the loads that change what the ranks hold.
        let data = Arc::new(worker::data_plane::DataPlane::new(
            data_ranks,
            worker::data_plane::Target {
                checkpoint: PathBuf::from(&launch.engine_config.model.model),
                experts: epoch.experts.clone(),
            },
            build.mem_fraction_static,
        ));
        // This node's schedulers, one thread per DP rank, each handing its batches to the workers
        // above through `worker::Batches`. A node that holds none keeps an empty list
        // and serves only what its peers' schedulers ask of it; a node that is the scheduler node
        // and holds none would accept requests and dispatch nothing, which is refused by name.
        let schedulers = if union.is_some() {
            let mut handles = Vec::with_capacity(workers.len());
            for (index, model_runner) in scheduled.iter().enumerate() {
                // A dedicated expert rank takes no request: it joins the steps the others open.
                if !model_runner.schedules() {
                    continue;
                }
                handles.extend(schedulers(
                    model_runner,
                    launch,
                    &models[index],
                    &workers[index..=index],
                    &inventories[index],
                    &memories[index],
                    connected.wake(),
                    &data,
                )?);
            }
            handles
        } else {
            schedulers(
                &first_plan,
                launch,
                &models[0],
                &workers,
                &inventories[0],
                &memories[0],
                connected.wake(),
                &data,
            )?
        };

        // The always-on counters: every stream a runner computes on is `device`, and the one the
        // host KV cache's pages move on is `copy`.
        let observed: Vec<(&'static str, platform::Stream)> = workers
            .iter()
            .flat_map(|worker| worker.streams().iter().cloned())
            .map(|stream| ("device", stream))
            .chain(
                workers
                    .iter()
                    .map(|worker| ("copy", worker.copy_stream().clone())),
            )
            .collect();
        let counters = worker::profiling::counters::Counters::start(
            &observed,
            !epoch.profiling.graph_pc_sampling,
        );
        let stats_reporters = (0..schedulers.len())
            .map(|_| scheduler::SchedulerStatsReporter::default())
            .collect();
        let mut process = Self {
            control: control
                .take()
                .context("a main process starts on the connection it opened")?,
            schedulers,
            union,
            stats_reporters,
            _counters: counters,
            control_calls: platform::cupti::blocking::Calls::default(),
            workers,
            memory: memories,
            storage: clients,
            blocking,
            modules,
            store: None,
            update,
            watchdog,
            programs,
            precopy: None,
            warming: Vec::new(),
            switchovers: 0,
            announced_at: None,
            reported_state: proto::request::wire::EngineState::Starting as i32,
            daemon_closed: false,
            announced: None,
            noted_process_closed: false,
            commands_seen: 0,
            command_wait: Wait::default(),
            warmup_wait: Wait::default(),
            pending_switch: None,
            warming_since: None,
            policy_probe_at: None,
            policy_missing_since: None,
            prepared: Vec::new(),
            retired: Vec::new(),
            generations,
            targets: model_runner.targets().to_vec(),
            program_server: model_runner.program_server().to_path_buf(),
            process,
            stopping: false,
            streams: Vec::new(),
            // Nothing is admitted until the epoch's first switchover, which is the boundary the
            // first batch is served at: before it the generation holds no kernel and no capture, so
            // a batch would find no graph. A standby parks itself here as well, and the `Promote`
            // that follows does not lift the boundary either.
            serving: false,
            boundary: false,
            pending_serve: false,
            serve_after_drain: None,
            admission_owed: false,
            max_running_requests: u32::try_from(build.max_running_requests.get())
                .unwrap_or(u32::MAX),
            catalogue: model_runner.prefill_catalogue(),
            serving_programs: Programs::new(),
            retired_programs: Programs::new(),
            prepared_programs: Programs::new(),
            standby: launch.standby(),
            promoted: false,
            parked: None,
            draining: None,
            paused: false,
            rollback: launch.rollback_s.map(Duration::from_secs),
            released: std::collections::BTreeSet::new(),
            data,
            owed: Vec::new(),
            weights: None,
            asleep: false,
        };
        if launch.standby() {
            process.park_standby()?;
        }
        // What the daemon waits for before the epoch's first `Precopy`: a process started beside a
        // serving one is `Standby`, and one launched to serve is `Ready`. Waiting to be ready until
        // this process had a policy would be waiting for the command that waits for it.
        let state = if launch.standby() {
            proto::request::wire::EngineState::Standby
        } else {
            proto::request::wire::EngineState::Ready
        };
        process.announce(state)?;
        Ok(process)
    }

    /// A standby that has prepared announces itself and executes nothing until promoted.
    ///
    /// Its schedulers hold no request and have served no epoch, so parking them would ask for a
    /// cache handover that belongs to a drained epoch; what makes this a park is that nothing is
    /// admitted. `Promote` is the command that ends it.
    ///
    /// The report means exactly "prepared and admitted nothing", which is what the daemon gates
    /// its promotion on: it is sent after `start` has mapped the weights, opened the memory and
    /// started the runners, and *before* the epoch's first generation has been warmed - a warmup
    /// belongs to the first switchover, and a standby's own generation reaches it when the daemon
    /// sends the epoch's programs.
    fn park_standby(&mut self) -> Result<()> {
        self.serving = false;
        // The launch generation, which is what `Parked` states: the daemon files a report by it
        // (`supervision/roster.rs:236-244` reads `launch_of(engine, parked.generation)`), and the
        // program generation is zero here - a standby parks before any `Precopy` names one - so a
        // report stating that would name a launch that does not exist and be dropped. The `Park`
        // command states the same number for the same reason.
        self.send_report(proto::process::Report {
            kind: Some(proto::process::report::Kind::Parked(
                proto::process::Parked {
                    generation: self.control.engine(),
                },
            )),
        });
        Ok(())
    }

    /// Stops every thread in dependency order and releases the owners.
    pub(super) fn close(&mut self) {
        // A shutdown finishes every request in flight as aborted, and the daemon learns that
        // from the request's own event - a connection that closes under it says only that the
        // process ended, not which request was lost with it.
        for served in std::mem::take(&mut self.streams) {
            let event = proto::request::wire::Event {
                request: served.request,
                kind: Some(proto::request::wire::event::Kind::Error(
                    proto::request::wire::Failed {
                        message: "the epoch was shut down with this request in flight".to_owned(),
                    },
                )),
            };
            self.send_event(
                served.front,
                proto::request::wire::from_engine::Kind::Event(event),
            );
        }
        // A rank that waits for its peers inside a step stops waiting: the peers are ending too.
        if let Some(union) = &self.union {
            union.end();
        }
        // The schedulers go first: their `Batches` holds a clone of every worker, and a device
        // thread cannot be joined while a scheduler could still submit to it.
        self.schedulers.clear();
        for worker in std::mem::take(&mut self.workers) {
            let Some(worker) = Arc::into_inner(worker) else {
                tracing::warn!(
                    "a device worker is still held after its schedulers ended; its thread ends \
                     when the last holder drops it"
                );
                continue;
            };
            if let Err(error) = worker.close() {
                tracing::warn!(%error, "closing a device worker failed");
            }
        }
        for error in self.blocking.close() {
            tracing::warn!(%error, "closing a blocking lane failed");
        }
        // The caches a park handed back name segments of the pools below: they go before the
        // resident owners close, so nothing the memory releases is still described as parked.
        self.parked = None;
        for memory in std::mem::take(&mut self.memory) {
            if let Err(error) = memory.close() {
                tracing::warn!(%error, "closing the resident owners failed");
            }
        }
        self.control.close();
    }
}

//! Starting and closing a main process: the control thread's connections, the ranks' load, the
//! device workers, the schedulers, and a standby's park; and the orderly close that ends them.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::update::HotUpdate;
use anyhow::{Context as _, Result, bail};
use build_server::Client;
use proto::config::EngineArgs;
use worker::blocking::{BlockingKind, ThreadPool};
use worker::control::{Control, Endpoints};
use worker::runners::products::GroupProducts;
use worker::watchdog::Watchdog;

use super::claim::claim;
use super::device_command;
use super::launch::{Epoch, ProcessInterface, handshake};
use super::ranks::{Backing, LoadedRank, Meeting, load_ranks};
use super::reports::{announce, state_failure};
use super::serve::Wait;
use super::{MainProcess, Peers};
use crate::modules::load::ModuleWork;
use crate::rank::ModelRunner;

impl MainProcess {
    /// Connects to the daemon, loads the epoch's model, and starts the epoch's threads in the order
    /// the architecture lists them (`docs/architecture.md`, The threads of a main process).
    ///
    /// The epoch's configuration is resolved against this process's launch first
    /// ([`ModelRunner`]), because three of the things the main process needs are the
    /// node's rather than the epoch's: which devices it serves, where its Build
    /// Server is, and what those devices measure.
    pub(super) fn start(launch: &EngineArgs, catalogue: &nn::family::Catalogue) -> Result<Self> {
        let build = &launch.engine_config;
        let family = crate::resolve(catalogue, Path::new(&launch.engine_config.model.model))?;
        let model_runner = ModelRunner::of(launch, family)?;
        let mut control = Control::connect(
            &Endpoints {
                requests: launch.requests.clone(),
            },
            launch.engine,
        )?;
        // The request interface's handshake: this process says who it is, and the
        // daemon answers with the epoch this node serves.
        // The epoch this rank belongs to is the daemon's, and it decides the storage this process
        // claims. A zero epoch names no epoch, and a world whose processes do not divide its ranks
        // between them is another deployment's: each is refused here, by name, rather than at the
        // claim or the first collective.
        if launch.epoch == 0 {
            bail!("the launch file states epoch 0, which names no epoch for this rank to serve");
        }
        if let Some(world) = &launch.world {
            if world.peers == 0 {
                bail!("the launch file states a world of several hosts with no peer main process");
            }
            if launch.scheduler_node != world.inherited.is_some() {
                bail!(
                    "the scheduler node's main process, and only it, inherits the world's reserved \
                     sockets: this one is{} the scheduler node",
                    if launch.scheduler_node { "" } else { " not" }
                );
            }
        } else if !launch.scheduler_node {
            bail!("a main process that holds every rank of its group holds its schedulers too");
        }
        let epoch = handshake(&mut control, launch)?;
        // The daemon's launch path blocks until this process states what it is, and it
        // sends the epoch's first generation command only once it has. `Starting` goes out
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
        // the generation reports, and every command of the generation layer - travels on this one.
        let mut process = ProcessInterface::connect(
            &launch.processes,
            &launch.engine_config.deployment,
            launch.engine,
        )?;
        // Every command of the generation layer arrives on this socket, and a stage waits for the
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
        // The epoch's other main processes, before any rank loads: the ranks' host group and their
        // communicators span every host, so each process must know the others are there first.
        let (meeting, links) = meet(launch)?;
        let storage = claim(
            &mut process,
            launch,
            &epoch.host,
            host_kv_cache,
            &rank_numbers,
        )?;
        let ranks = load_ranks(
            rank_runners,
            Backing::Serving(&storage),
            launch,
            &epoch,
            meeting,
        )?;

        // The resident owners are opened before any thread that borrows them, and
        // released in reverse order by `close`. The weights are mapped before the capacity plan
        // measures, so the plan sees a device with the model already in it - the figure it computes
        // is what is left, not what was there before the load. Pools sized first and weights after
        // would over-commit the device by exactly the model's size.
        let blocking = ThreadPool::new(model_runner.blocking_limits());
        let modules = blocking.lane::<ModuleWork>(BlockingKind::ModuleLoad)?;
        let packages = Arc::new(Mutex::new(Client::connect(model_runner.build_server())?));

        // The watchdog exists before any device command: `start`'s own commands
        // are bounded by the same stall bound as every command the control thread
        // issues afterwards.
        let watchdog = Watchdog::new(
            launch
                .watchdog_timeout()
                .unwrap_or_else(|| model_runner.watchdog_timeout()),
        )?;
        let mut update = HotUpdate::new(epoch.generation);

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
        // The worker group of each device, in the order `workers` holds them.
        let mut groups: Vec<String> = Vec::with_capacity(ranks.len());
        let placed_groups: Vec<String> = ranks
            .iter()
            .map(|rank| rank.model_runner.group().to_owned())
            .collect();
        for (rank_index, mut rank) in ranks.into_iter().enumerate() {
            // The ranks of its group this process holds, which the scheduler counts as one.
            let group_ranks = placed_groups
                .iter()
                .filter(|group| **group == placed_groups[rank_index])
                .count();
            let spawned = rank.spawn_workers(
                (rank_index, group_ranks),
                &group_products,
                epoch.generation,
                &blocking,
            )?;
            let LoadedRank {
                model_runner: rank_runner,
                weights,
                client,
                model,
                memory,
                inventory,
                ..
            } = rank;
            let client = client.context("a serving rank holds its Storage Server client")?;
            for (device, worker, generation) in spawned {
                // The precopy shadows a replaced operator beside the serving one, so the region it
                // runs in is mapped before any batch: the worker reserves it from the scratch pool
                // the graph storage sized. Nothing is serving yet, so the control thread waits here;
                // `serve` never does.
                //
                // The warmup is not here: the epoch's first generation holds no kernel until the
                // daemon's precopy fills it - and a precopy that names no kernel (the startup
                // switch) fills it from the store's built packages - so there is no bucket to warm
                // yet. It runs at the first switchover, which is the boundary the first request is
                // served at.
                let shadow = worker.set_shadow()?;
                device_command(&watchdog, device, "shadow region", shadow)
                    .with_context(|| format!("mapping the shadow region of device {}", device.0))?;
                // The park's and the wake's commands reach the Storage Server's quota from the
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
                data_ranks.push(crate::data_plane::Rank {
                    rank: rank_runner.rank(),
                    worker: Arc::downgrade(&worker),
                    device,
                    storage: Arc::clone(&client),
                    cache: memory.shared_cache(),
                    model: Arc::new(crate::rank::RankModel::new(rank_runner.clone())),
                    serving: weights.clone(),
                    host: epoch.host.clone(),
                });
                workers.push(worker);
                groups.push(rank_runner.group().to_owned());
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
        let data = Arc::new(crate::data_plane::DataPlane::new(
            data_ranks,
            crate::data_plane::Target {
                checkpoint: PathBuf::from(&launch.engine_config.model.model),
                experts: epoch.experts.clone(),
            },
            build.mem_fraction_static,
        ));
        // Each device with the worker group its rank serves in, which routes a batch to it.
        let placed: Vec<(String, Arc<worker::device::Worker>)> = groups
            .iter()
            .cloned()
            .zip(workers.iter().map(Arc::clone))
            .collect();
        // The links to the epoch's other main processes, over the channels `meet` opened: on the
        // scheduler node one per peer, which the schedulers' batches go through, and on a peer the
        // one that runs the scheduler node's batches on this process's workers.
        let peers = match links {
            Links::Alone => Peers::Alone,
            Links::Accepted(channels) => {
                let liveness = worker::peers::Liveness::new(connected.wake());
                let remotes = channels
                    .into_iter()
                    .map(|(rank, channel)| {
                        worker::peers::Remote::spawn(rank, channel, Arc::clone(&liveness))
                    })
                    .collect::<worker::Result<Vec<_>>>()?;
                Peers::Scheduler { remotes, liveness }
            }
            Links::Registered(channel) => {
                let liveness = worker::peers::Liveness::new(connected.wake());
                let ranks = Arc::new(worker::batch::Workers::shared(placed.clone())?);
                // The scheduler node plans a group whose first rank is this process's against
                // that rank's report.
                let report =
                    super::ranks::stated_report(&inventories[0], &memories[0], &data, false)?;
                Peers::Peer {
                    upstream: worker::peers::Upstream::serve(
                        channel,
                        ranks,
                        report,
                        Arc::clone(&liveness),
                    )?,
                    liveness,
                }
            }
        };
        let remotes = match &peers {
            Peers::Scheduler { remotes, .. } => remotes.clone(),
            Peers::Alone | Peers::Peer { .. } => Vec::new(),
        };
        // This node's schedulers, one thread per DP rank, each handing its batches to the workers
        // above through `worker::Batches`. A node that holds none keeps an empty list
        // and serves only what the scheduler node's batches ask of it; a node that is the scheduler
        // node and holds none would accept requests and dispatch nothing, which is refused by name.
        let schedulers = if !launch.scheduler_node {
            Vec::new()
        } else if union.is_some() {
            let mut handles = Vec::with_capacity(workers.len());
            for (index, model_runner) in scheduled.iter().enumerate() {
                // A dedicated expert rank takes no request: it joins the steps the others open.
                if !model_runner.schedules() {
                    continue;
                }
                let host_cache = model_runner.host_kv_cache(&models[index]);
                let pool = super::ranks::Pool {
                    group: groups[index].clone(),
                    report: super::ranks::stated_report(
                        &inventories[index],
                        &memories[index],
                        &data,
                        host_cache,
                    )?,
                };
                handles.push(super::ranks::scheduler(
                    model_runner,
                    launch,
                    &models[index],
                    vec![placed[index].clone()],
                    Vec::new(),
                    vec![pool],
                    host_cache,
                    connected.wake(),
                    &data,
                )?);
            }
            handles
        } else {
            let blocks = super::ranks::Blocks {
                first: launch.rank as usize,
                groups: &groups,
                inventories: &inventories,
                memories: &memories,
                remotes: &remotes,
                data: &data,
            };
            let mut handles = Vec::new();
            for block in blocks.of(&first_plan, &models[0], launch)? {
                handles.push(super::ranks::scheduler(
                    &first_plan,
                    launch,
                    &models[0],
                    placed
                        .iter()
                        .filter(|(group, _)| block.groups.contains(group))
                        .cloned()
                        .collect(),
                    block.remotes,
                    block.pools,
                    block.host_cache,
                    connected.wake(),
                    &data,
                )?);
            }
            handles
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
        update.generations = generations;
        // The route instance is held to the ceilings of the batch the schedulers were built for,
        // as theirs are.
        let router = scheduler::Router::new(
            schedulers
                .first()
                .map(|handle| scheduler::ceilings_of(scheduler::BatchCapacity::of(handle.info())))
                .unwrap_or_default(),
        );
        let mut process = Self {
            control: control
                .take()
                .context("a main process starts on the connection it opened")?,
            peers,
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
            update,
            watchdog,
            packages,
            precopy: None,
            warming: Vec::new(),
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
            warming_steps: Vec::new(),
            policy_answers: Vec::new(),
            router,
            route_prepared: None,
            targets: model_runner.targets().to_vec(),
            build_server: model_runner.build_server().to_path_buf(),
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
            standby: launch.standby(),
            promoted: false,
            standby_storage: launch.standby_storage,
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
    /// sends the epoch's packages.
    fn park_standby(&mut self) -> Result<()> {
        self.serving = false;
        // The launch generation, which is what `Parked` states: the daemon files a report by it
        // (`supervision/roster.rs:236-244` reads `launch_of(engine, parked.generation)`), and the
        // module generation is zero here - a standby parks before any `Precopy` names one - so a
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
        // process ended, not which request was lost with it. An epoch that lost another of its
        // main processes is a loss and not a shutdown: its requests are left to the close of this
        // connection, which the daemon reads as the engine's loss and migrates them on.
        let streams = std::mem::take(&mut self.streams);
        let lost = self.peers.lost().is_some();
        for served in streams.into_iter().filter(|_| !lost) {
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
        // The peer links close first: a scheduler waiting for a peer's answer is answered by the
        // close, and the peers read it as this process's end. The schedulers go next: their
        // `Batches` holds a clone of every worker, and a device thread cannot be joined while a
        // scheduler could still submit to it.
        self.peers.close();
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

/// How long the scheduler node waits for every peer main process to register, and a peer for the
/// scheduler node to accept it: each starts at once and registers before it loads anything.
const PEER_START: Duration = Duration::from_secs(300);

/// The channels to the epoch's other main processes, as [`meet`] opened them.
enum Links {
    /// This process holds every rank of its group.
    Alone,
    /// The scheduler node's: every peer's registered channel, by the first rank it serves.
    Accepted(Vec<(u32, ipc::Channel)>),
    /// A peer's: its channel to the scheduler node.
    Registered(ipc::Channel),
}

/// Meets the epoch's other main processes the launch names: the scheduler node accepts every
/// peer's registration on the listener it inherited, and a peer dials it and registers. Answers
/// where the group's ranks form their host group, and the channels.
///
/// # Errors
///
/// Fails when a peer does not register in time, or registers for another epoch or build.
fn meet(launch: &EngineArgs) -> Result<(Meeting, Links)> {
    let Some(world) = &launch.world else {
        return Ok((Meeting::Local, Links::Alone));
    };
    match world.inherited {
        Some(inherited) => {
            // SAFETY: the daemon reserved both sockets for this process and its supervisor
            // placed them at these descriptors (`Spawn::inherit`); nothing else in the process
            // owns them.
            let (channel, store) = unsafe {
                (
                    <std::net::TcpListener as std::os::fd::FromRawFd>::from_raw_fd(
                        inherited.channel,
                    ),
                    <std::net::TcpListener as std::os::fd::FromRawFd>::from_raw_fd(inherited.store),
                )
            };
            let listener = ipc::Listener::from_tcp(channel)
                .context("taking the peer listener this process inherited")?;
            tracing::info!(
                peers = world.peers,
                channel = %world.channel,
                "awaiting the epoch's peer main processes"
            );
            let accepted =
                worker::peers::accept(&listener, world.peers as usize, launch.engine, PEER_START)?;
            Ok((
                Meeting::World {
                    address: world.store.clone(),
                    listener: Some(store),
                },
                Links::Accepted(accepted),
            ))
        }
        None => {
            let channel =
                worker::peers::register(&world.channel, launch.rank, world.scheduler, PEER_START)?;
            tracing::info!(
                rank = launch.rank,
                scheduler = %world.channel,
                "registered with the epoch's scheduler node"
            );
            Ok((
                Meeting::World {
                    address: world.store.clone(),
                    listener: None,
                },
                Links::Registered(channel),
            ))
        }
    }
}

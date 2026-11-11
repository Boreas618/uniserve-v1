//! Loading this process's ranks, one thread per rank: the collective groups they agree over, each
//! rank's model, weights and resident owners ([`LoadedRank`]), and the schedulers the scheduler
//! node builds over the loaded ranks' workers.

use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use ops::Generation;
use proto::config::EngineArgs;
use worker::blocking::ThreadPool;
use worker::device::Worker;
use worker::memory::Memory;
use worker::memory::retained::{LocalStorage, Retained};
use worker::memory::weights::WeightStore;
use worker::parallel::host::{HostGroup, Rendezvous, STARTUP_TIMEOUT};
use worker::runners::products::{GroupProducts, RankProducts};

use super::launch::Epoch;
use crate::rank::{ModelRunner, Weights};

/// What the ranks of a start load onto.
#[derive(Clone, Copy)]
pub(super) enum Backing<'a> {
    /// The node's Storage Server, whose claim the daemon answered: each rank publishes its weights
    /// there and maps its resident memory from it, and its parallel groups form their NCCL
    /// communicators.
    Serving(&'a ipc::StorageLaunch),
    /// Nothing outside this process: a recorded start ([`crate::record`]) loads each weight's
    /// element type and extents from the shard headers alone, maps its resident memory as storage
    /// of its own, and its parallel groups state their split with no communicator, since a
    /// recorded walk issues no collective.
    Recording,
}

/// One rank of this process, loaded: what its worker is built from.
pub(super) struct LoadedRank {
    /// The rank's own model runner, whose one device is the rank's.
    pub(super) model_runner: ModelRunner,
    /// The weights its load reached, in order: what a weight update pairs against.
    pub(super) weights: Vec<worker::memory::weights::LoadedWeight>,
    /// The rank's Storage Server client, whose scope is the rank's; a recorded start has none.
    pub(super) client: Option<Arc<Retained>>,
    /// The rank's partition of the model, mapped on its device.
    pub(super) model: crate::rank::Loaded,
    /// The rank's resident owners.
    pub(super) memory: Memory,
    /// What the rank reports of itself.
    pub(super) inventory: ipc::MainProcessInfo,
    /// The rank's graph storage figures.
    pub(super) graph: worker::device::GraphStorage,
    /// The collective groups its runners issue on.
    pub(super) groups: crate::rank::RankGroups,
}

impl LoadedRank {
    /// Builds this rank's runners on each of its devices and starts each device's worker thread,
    /// whose start records the runners' walk ([`Worker::recorded`]) and claims each stream's arena
    /// region from it. Answers each device with its worker and the epoch's first generation, which
    /// holds no kernel yet.
    ///
    /// `(index, count)` is this rank's place among the process's ranks and the count of its worker
    /// group's ranks this process holds, which the scheduler sees as one pool, `products`
    /// the products the ranks share, `generation` the epoch's program generation and `blocking` the
    /// process's lanes. A serving start and a recorded start build their workers here alike, so
    /// what a recorded start reports is what a serving one launches.
    pub(super) fn spawn_workers(
        &mut self,
        (index, count): (usize, usize),
        products: &Arc<GroupProducts>,
        generation: u64,
        blocking: &ThreadPool,
    ) -> Result<Vec<(platform::DeviceId, Worker, Arc<Generation>)>> {
        // The scheduler sees the group as one pool, and each rank encodes its own units, so the
        // pool holds every rank's host tasks.
        self.inventory.host_task_capacity *= u32::try_from(count).unwrap_or(u32::MAX);
        // The device's one pool set is the resident owners' allocator: it is opened with the memory
        // and handed to the worker thread, so every owner and every runner books against the same
        // allowance.
        let pools = self.memory.pools();
        let products: Arc<dyn worker::runners::products::Products> =
            Arc::new(RankProducts::new(Arc::clone(products), index));
        // What the rank's trajectories are computed with, which keys the pages of its host latent
        // pool: the weights it loaded.
        let identity = crate::data_plane::identity_of(&self.weights);
        let mut spawned = Vec::with_capacity(self.model_runner.devices().len());
        for device in self.model_runner.devices() {
            let generation = Arc::new(Generation::new(generation, std::iter::empty()));
            // The scheduler counts this process's ranks as one, the first rank's report, so the
            // first rank states the host latent pool for all of them.
            let runners = self.model_runner.runners(
                *device,
                &self.model,
                &mut self.memory,
                &generation,
                blocking,
                &self.groups,
                self.client.as_ref(),
                &identity,
                (index == 0).then_some(&mut self.inventory),
            )?;
            // The scratch pool's figure comes from the one call that sized it, with the memory: the
            // shadow region and the state group's snapshot are one reservation, and a second
            // computation here could disagree with the pool the allocator reserved. The shadow
            // region alone is what the twins allocate.
            let worker = Worker::spawn(
                *device,
                runners,
                self.model_runner.events(),
                Arc::clone(&pools),
                Arc::clone(&products),
                self.graph.scratch,
                self.graph.shadow,
            )?;
            spawned.push((*device, worker, generation));
        }
        Ok(spawned)
    }
}

/// Where the ranks of a process's worker group form their host group.
pub(super) enum Meeting {
    /// This process holds every rank: they meet over the loopback.
    Local,
    /// The group has ranks on other hosts: its members meet at `address`, which the process that
    /// holds rank 0 serves on `listener`, the socket its daemon reserved; a peer holds none.
    World {
        address: String,
        listener: Option<std::net::TcpListener>,
    },
}

/// Loads every rank of this process, each on a thread of its own.
///
/// The ranks agree their capacity and exchange their communicators' ids over one host group, whose
/// collectives every member must reach, so they run side by side rather than one after another; the
/// weights of four ranks then also load at once. The group is the one the capacity plan agrees
/// over: formed over the loopback when this process holds all its members, and at the group's
/// rendezvous ([`Meeting::World`]) when other hosts hold some, each rank a member under its rank
/// number in the group.
pub(super) fn load_ranks(
    model_runners: Vec<ModelRunner>,
    backing: Backing<'_>,
    launch: &EngineArgs,
    epoch: &Epoch,
    meeting: Meeting,
) -> Result<Vec<LoadedRank>> {
    let count = model_runners.len();
    // The members' numbers: this process's place among them, and how many there are.
    let (first, members, address, mut listener) = match meeting {
        Meeting::Local => {
            let listener = std::net::TcpListener::bind(("127.0.0.1", 0))
                .context("binding the rendezvous of this process's ranks")?;
            let address = listener
                .local_addr()
                .context("reading the rendezvous of this process's ranks")?
                .to_string();
            (0, count, address, Some(listener))
        }
        Meeting::World { address, listener } => {
            let first = launch.rank as usize;
            let members = launch.world_size as usize;
            if first + count > members {
                bail!(
                    "this process serves ranks {first}..{} of a group of {members}",
                    first + count
                );
            }
            if (first == 0) != listener.is_some() {
                bail!(
                    "the process that holds rank 0 serves the group's rendezvous, and only it: \
                     this one holds rank {first} and {} the socket",
                    if listener.is_some() {
                        "inherited"
                    } else {
                        "was given no"
                    }
                );
            }
            (first, members, address, listener)
        }
    };
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(count);
        for (index, model_runner) in model_runners.into_iter().enumerate() {
            let rendezvous = match listener.take() {
                Some(listener) => Rendezvous::serve(address.clone(), listener),
                None => Rendezvous::connect(address.clone()),
            };
            let handle = std::thread::Builder::new()
                .name(format!("rank-load-{index}"))
                .spawn_scoped(scope, move || {
                    load_rank(
                        model_runner,
                        rendezvous,
                        first + index,
                        members,
                        backing,
                        launch,
                        epoch,
                    )
                })
                .context("starting a rank's load")?;
            handles.push(handle);
        }
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("a rank's load panicked"))?
            })
            .collect()
    })
}

/// Joins the group as `member` of its `members`, opens the rank's collectives, loads its partition
/// and opens its memory.
fn load_rank(
    model_runner: ModelRunner,
    rendezvous: Rendezvous,
    member: usize,
    members: usize,
    backing: Backing<'_>,
    launch: &EngineArgs,
    epoch: &Epoch,
) -> Result<LoadedRank> {
    // A process that succeeds an epoch maps the cache segments its predecessor serves from: a
    // standby takes them over when it is promoted, and a switch's target serves from them under the
    // predecessor's epoch, as the target's own planes would otherwise need a second cache's storage
    // beside the serving epoch and its standby. Only a standby keys its cache by the predecessor's
    // epoch, because only a standby becomes that epoch's process.
    let adopted = epoch
        .predecessor_cache
        .iter()
        .find(|cache| cache.worker_group == model_runner.group() && !cache.segments.is_empty())
        .map(|cache| ipc::CacheTable {
            segments: cache
                .segments
                .iter()
                .map(|segment| ipc::CacheSegment {
                    epoch: segment.epoch,
                    ordinal: segment.ordinal,
                    units: segment.units,
                })
                .collect(),
            held: cache.held,
        });
    let model_runner = model_runner.sharing_cache_of(
        epoch.predecessor_epoch.filter(|_| launch.standby()),
        adopted,
    );
    let host = HostGroup::connect(rendezvous, member, members, STARTUP_TIMEOUT)
        .context("joining this process's ranks")?;
    // The rank's Storage Server and its claim are the node's, which the daemon answered for; the
    // retained weights outlive the epoch, so the store is built once, before the model loads,
    // and keeps what it maps for the process's life.
    let client = match backing {
        Backing::Serving(storage) => Some(
            Retained::connect(
                storage,
                &launch.group,
                model_runner.rank(),
                epoch.generation,
                epoch.predecessor_epoch,
            )
            .context("connecting to the Storage Server")?,
        ),
        Backing::Recording => None,
    };
    let communicates = client.is_some();

    // A weight's region is one device's address, so the store maps on the device this rank's model
    // is loaded for; the client is shared, the mapping is not.
    let device = *model_runner
        .devices()
        .first()
        .context("this rank serves no device to map its weights on")?;
    let shape = model_runner.parallel_shape()?;

    // NCCL keeps its own device buffers, so a communicator opens before the capacity plan measures
    // what is left of the device. A media rank's groups span the host group; a text rank's tensor
    // group is its worker group's ranks, one span of the host group among the deployment's groups.
    let whole = |size: usize, rank: usize| {
        if size > 1 && size != members {
            bail!(
                "a parallel group of {size} ranks in a host group of {members}: the group's ranks \
                 form one parallel group, and a deployment of several such groups is not served"
            );
        }
        Ok(crate::rank::Collective {
            spans: vec![(0, size)],
            member: Some((0, rank)),
        })
    };
    let tensor = match model_runner.tensor_spans()? {
        Some(collective) => collective,
        None => whole(shape.tensor, shape.tensor_rank)?,
    };
    let groups = crate::rank::RankGroups {
        sequence: group_over(
            &host,
            device,
            &whole(shape.sequence, shape.sequence_rank)?,
            communicates,
        )
        .context("forming the sequence group")?,
        tensor: group_over(&host, device, &tensor, communicates)
            .context("forming the tensor group")?,
    };

    // The load and the capacity plan are one call because the order between them is the plan's: the
    // weights are mapped first, so what the plan measures is what is left of the device rather than
    // what was there before the model.
    let ((model, memory, mut inventory, graph), weights) = match &client {
        Some(client) => {
            let mut weights = WeightStore::new(Arc::clone(client), device);
            let loaded = model_runner
                .load_and_open(
                    Weights::Retained(&mut weights),
                    &mut &**client,
                    &host,
                    &epoch.experts,
                )
                .context("loading the epoch's model")?;
            let log = weights
                .finish()
                .context("publishing the granules of the epoch's packed weights")?;
            (loaded, log)
        }
        // No weight is mapped, so the plan sees the whole device: a recorded start sizes its cache
        // larger than a served one wherever the deployment does not bound it.
        None => {
            let loaded = model_runner
                .load_and_open(Weights::Meta, &mut LocalStorage, &host, &epoch.experts)
                .context("loading the epoch's model from its checkpoint's headers")?;
            (loaded, Vec::new())
        }
    };
    // Where another host's Transfer Server reads this rank's planes from, for a prefill/decode
    // hand-off between groups of two hosts: this host's Storage Server endpoint, which a
    // deployment of several nodes states.
    if let (Backing::Serving(storage), Some(cache)) = (backing, inventory.prefix_cache.as_mut()) {
        cache.storage_endpoint.clone_from(&storage.tcp);
    }
    tracing::info!(
        rank = model_runner.rank(),
        group = model_runner.group(),
        "the rank loaded its model"
    );
    Ok(LoadedRank {
        model_runner,
        weights,
        client,
        model,
        memory,
        inventory,
        graph,
        groups,
    })
}

/// The collective group of `collective` this rank is a member of, absent for a group of one.
///
/// A group that `communicates` forms its NCCL communicator: each group's first member draws the
/// group's id and the host group carries it (`parallel`'s statement of what that channel is for),
/// one broadcast per group of more than one member, which every member of the host group takes part
/// in, in the order the spans state; `Communicator::init` then blocks until every member of the
/// rank's own group joined. One that does not states the split alone ([`nn::Group::recording`]).
fn group_over(
    host: &HostGroup,
    device: platform::DeviceId,
    collective: &crate::rank::Collective,
    communicates: bool,
) -> Result<Option<Arc<nn::Group>>> {
    let members = host.size();
    for &(first, size) in &collective.spans {
        if size > 1 && first + size > members {
            bail!(
                "a parallel group of {size} ranks from member {first} in a host group of \
                 {members}: a group's ranks are members of the host group"
            );
        }
    }
    let mine = collective.member.filter(|(span, _)| {
        collective
            .spans
            .get(*span)
            .is_some_and(|(_, size)| *size > 1)
    });
    if !communicates {
        return mine
            .map(|(span, rank)| {
                let size = collective.spans[span].1;
                Ok(Arc::new(nn::Group::recording(rank as u64, size as u64)?))
            })
            .transpose();
    }
    let mut own = None;
    for (index, &(first, size)) in collective.spans.iter().enumerate() {
        if size <= 1 {
            continue;
        }
        let root = host.rank() == first;
        let id = if root {
            Some(platform::UniqueId::new()?)
        } else {
            None
        };
        let bytes = host.broadcast_bytes(id.as_ref().map(|id| &id.0[..]), first)?;
        if mine.is_some_and(|(span, _)| span == index) {
            own = Some(platform::UniqueId(bytes.try_into().map_err(
                |bytes: Vec<u8>| {
                    anyhow::anyhow!("a communicator's id is 128 bytes, not {}", bytes.len())
                },
            )?));
        }
    }
    let (Some((span, rank)), Some(id)) = (mine, own) else {
        return Ok(None);
    };
    let size = collective.spans[span].1;
    let size = i32::try_from(size).context("a group wider than NCCL addresses")?;
    let member = i32::try_from(rank).context("a rank past what NCCL addresses")?;
    let communicator = platform::Communicator::init(device, size, member, &id)
        .context("joining a communicator")?;
    Ok(Some(Arc::new(nn::Group::new(communicator, size as u64)?)))
}

/// What a rank states of itself to the scheduler that drives its group, from `rank`, the report its
/// capacity plan made: the identities every cache level keys its pages by (`storage::key::page_seed`,
/// from the weights the process serves and the pages' layout), and with `host_cache` the bytes one
/// token takes in its host KV cache.
///
/// The scheduler node states its own ranks with this, and a peer states its first rank with it
/// when the scheduler node asks ([`worker::peers::Remote::info`]).
pub(super) fn stated_report(
    rank: &ipc::MainProcessInfo,
    memory: &Memory,
    data: &crate::data_plane::DataPlane,
    host_cache: bool,
) -> Result<ipc::MainProcessInfo> {
    let mut rank = rank.clone();
    if let Some(cache) = rank.prefix_cache.as_mut() {
        cache.identities = memory.page_identities(&data.serving_identity()?);
        if host_cache {
            cache.host_token_bytes = memory.host_token_bytes();
        }
    }
    Ok(rank)
}

/// One worker group a scheduler drives: its name, and the report of its first rank, which the
/// scheduler plans the group against ([`stated_report`]).
pub(super) struct Pool {
    pub(super) group: String,
    pub(super) report: ipc::MainProcessInfo,
}

/// One scheduler thread over `pools`, the worker groups of one data-parallel rank.
///
/// The scheduler forms a batch for one of its groups, every rank of that group issues it, and the
/// scheduler reads the result - nothing crosses threads except the batch and its result. Its batches
/// go to `devices`, this node's devices with the group each is a rank of, and to `remotes`, the peer
/// links of the processes that hold ranks of these groups on other hosts.
///
/// `model_runner` is a rank of this process as the scheduler plans it ([`ModelRunner::scheduled`]),
/// `model` its loaded model: every rank of a deployment is built from one configuration, so its
/// bounds are every group's. `host_cache` says whether the scheduler keeps a host KV cache, which
/// it does for a group of one device of this node.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument has its own owner in the process - the launch, the loaded model, the \
              workers, the peer links, the pools, the wake and the data plane - and a struct \
              gathering them would bundle unrelated state for one call"
)]
pub(super) fn scheduler(
    model_runner: &ModelRunner,
    launch: &EngineArgs,
    model: &crate::rank::Loaded,
    devices: Vec<(String, Arc<Worker>)>,
    remotes: Vec<(Vec<String>, Arc<worker::peers::Remote>)>,
    pools: Vec<Pool>,
    host_cache: bool,
    control: ipc::Wake,
    data: &Arc<crate::data_plane::DataPlane>,
) -> Result<scheduler::SchedulerHandle> {
    // One scheduler per scheduler thread: it reads the engine configuration as this rank resolves
    // it, with the model's facts, and the limits this rank captured.
    let (args, limits) = model_runner.scheduler_args(launch, model, host_cache)?;

    // The scheduler plans against a description of each group, and a group is one rank to it: its
    // ranks run each batch together, its first rank's report states what they hold, its capacities
    // are the launch's (the rank reports what it holds, not what it was launched for), and its
    // longest sequence is the model's bound. The inventory is the worker's to hold and the boot's
    // to state, because neither number can be derived from a batch.
    let inventory = pools
        .into_iter()
        .map(|pool| ipc::execution::Inventory {
            layout: ipc::execution::PoolLayout {
                queue_depth: model_runner.queue_depth(),
                ranks: model_runner.world_size(),
                // Every component the rank reports is placed on the group's ranks. A text rank
                // reports the one default component, and a media rank one per block of its
                // pipeline, which the scheduler reads a call's width and owner from.
                stages: pool
                    .report
                    .stages
                    .iter()
                    .flat_map(|component| {
                        scheduler::WorkerGroupConfig::single_component(
                            &component.name,
                            model_runner.world_size(),
                        )
                    })
                    .collect(),
            },
            group: pool.group,
            ranks: vec![pool.report],
            max_model_len: model_runner.engine_config().model.max_model_len,
        })
        .collect();
    // The scheduler submits through the same devices this process drives, so a batch and a
    // switchover reach one worker thread per device and never two.
    let mut stated = worker::batch::Workers::shared(devices)?
        .stating(inventory)
        .with_remotes(remotes);
    if host_cache {
        stated = stated.keeping_host_cache(worker::batch::HostCacheEndpoint { endpoint: None });
    }
    let batches: Arc<dyn ipc::execution::Batches> = Arc::new(crate::data_plane::RankBatches {
        workers: stated,
        data: Arc::clone(data),
    });

    // The control thread forwards what a scheduler produced, so it sleeps until a step says there
    // is something: a request's tokens reach the daemon when they exist, not at the next pass.
    let progress = scheduler::CommandWaker::new(move || control.wake());
    scheduler::SchedulerHandle::over_batches(&args, limits, batches, progress)
}

/// How long the scheduler node waits for a peer to report the first rank of a worker group it
/// holds: the peer answers once its ranks loaded, which the host group's capacity agreement has
/// just shown they did, so this bounds only the peer's start of its workers.
const PEER_REPORT: std::time::Duration = std::time::Duration::from_secs(300);

/// What the scheduler node forms its schedulers from: this process's ranks and its links to the
/// epoch's other main processes.
pub(super) struct Blocks<'a> {
    /// The deployment rank of this process's first rank.
    pub(super) first: usize,
    /// The worker group of each of this process's ranks, in rank order.
    pub(super) groups: &'a [String],
    /// What each of this process's ranks reports of itself, in rank order.
    pub(super) inventories: &'a [ipc::MainProcessInfo],
    /// Each rank's resident owners, in rank order.
    pub(super) memories: &'a [Memory],
    /// The peer links, one per other main process of the epoch.
    pub(super) remotes: &'a [Arc<worker::peers::Remote>],
    /// The process's data plane.
    pub(super) data: &'a Arc<crate::data_plane::DataPlane>,
}

/// One scheduler's share of the epoch: the worker groups it drives, each with its first rank's
/// report, and the peer links of the processes that hold their ranks on other hosts.
pub(super) struct Block {
    pub(super) groups: Vec<String>,
    pub(super) pools: Vec<Pool>,
    pub(super) remotes: Vec<(Vec<String>, Arc<worker::peers::Remote>)>,
    /// Whether the scheduler keeps a host KV cache: it does for one group of one device of this
    /// node, whose pages its copy stream moves.
    pub(super) host_cache: bool,
}

impl Blocks<'_> {
    /// The schedulers of a deployment whose ranks `model_runner` and `model` are one of: for a
    /// text deployment one per data-parallel rank, over its block of the deployment's serving
    /// groups ([`scheduler::WorkerGroupConfig::scheduler_blocks`]) - a prefill/decode deployment's
    /// two groups are one scheduler's - and for a media deployment one over the process's group,
    /// which its blocks' groups form together.
    ///
    /// # Errors
    ///
    /// Fails when the process holds no device, when the serving groups do not form the
    /// deployment's data-parallel blocks, or when a group's first rank cannot be reported: a peer
    /// reports only its process's first rank, so a group whose first rank lies on another host
    /// starts that host's run of ranks.
    pub(super) fn of(
        &self,
        model_runner: &ModelRunner,
        model: &crate::rank::Loaded,
        launch: &EngineArgs,
    ) -> Result<Vec<Block>> {
        if self.groups.is_empty() {
            bail!("the scheduler node holds no device to serve a batch on");
        }
        let config = &launch.engine_config;
        let deployment = &config.parallel.worker_groups;
        let text = matches!(model, crate::rank::Loaded::Text(_));
        let blocks: Vec<Vec<String>> = if text {
            scheduler::WorkerGroupConfig::scheduler_blocks(
                deployment,
                config.parallel.data_parallel_size.get(),
            )?
            .into_iter()
            .map(|block| {
                block
                    .into_iter()
                    .map(|index| deployment[index].id.0.clone())
                    .collect()
            })
            .collect()
        } else {
            vec![vec![launch.group.clone()]]
        };
        // Each peer link with the groups its process holds ranks of: a text group with a rank on
        // the peer's host, and the process's one group for a media deployment.
        let tagged: Vec<(Vec<String>, Arc<worker::peers::Remote>)> = self
            .remotes
            .iter()
            .map(|remote| {
                let held = if text {
                    groups_on_host_of(deployment, remote.rank() as usize)
                } else {
                    vec![launch.group.clone()]
                };
                (held, Arc::clone(remote))
            })
            .collect();
        let firsts = scheduler::WorkerGroupConfig::first_ranks(deployment);
        let mut formed = Vec::with_capacity(blocks.len());
        for groups in blocks {
            let local_devices = self
                .groups
                .iter()
                .filter(|group| groups.contains(group))
                .count();
            let remotes: Vec<(Vec<String>, Arc<worker::peers::Remote>)> = tagged
                .iter()
                .filter(|(held, _)| held.iter().any(|group| groups.contains(group)))
                .map(|(held, remote)| {
                    let mine = held
                        .iter()
                        .filter(|group| groups.contains(group))
                        .cloned()
                        .collect();
                    (mine, Arc::clone(remote))
                })
                .collect();
            let host_cache = model_runner.host_kv_cache(model)
                && groups.len() == 1
                && local_devices == 1
                && remotes.is_empty();
            let mut pools = Vec::with_capacity(groups.len());
            for group in &groups {
                let report = if text {
                    let index = deployment
                        .iter()
                        .position(|stated| stated.id.0 == *group)
                        .with_context(|| format!("the deployment places no group {group}"))?;
                    self.report_of(firsts[index], host_cache)
                        .with_context(|| format!("reporting worker group {group}"))?
                } else {
                    stated_report(
                        &self.inventories[0],
                        &self.memories[0],
                        self.data,
                        host_cache,
                    )?
                };
                pools.push(Pool {
                    group: group.clone(),
                    report,
                });
            }
            formed.push(Block {
                groups,
                pools,
                remotes,
                host_cache,
            });
        }
        Ok(formed)
    }

    /// The report of the deployment's rank `rank`: this process's own rank, or the first rank of
    /// a peer's process, which the peer answers.
    fn report_of(&self, rank: usize, host_cache: bool) -> Result<ipc::MainProcessInfo> {
        if let Some(local) = rank
            .checked_sub(self.first)
            .filter(|local| *local < self.inventories.len())
        {
            return stated_report(
                &self.inventories[local],
                &self.memories[local],
                self.data,
                host_cache,
            );
        }
        let remote = self
            .remotes
            .iter()
            .find(|remote| remote.rank() as usize == rank)
            .with_context(|| {
                format!(
                    "rank {rank} starts a worker group and no main process of the epoch starts its \
                     ranks there: a peer reports its first rank, so a group's ranks on another \
                     host begin that host's ranks"
                )
            })?;
        Ok(remote.info(PEER_REPORT)?)
    }
}

/// The groups of `groups` with a rank on the host of the deployment's rank `rank`.
fn groups_on_host_of(groups: &[scheduler::WorkerGroupConfig], rank: usize) -> Vec<String> {
    let Some((group, position)) = scheduler::WorkerGroupConfig::place_rank(groups, rank) else {
        return Vec::new();
    };
    let host = &groups[group].ranks[position].host;
    groups
        .iter()
        .filter(|group| group.ranks.iter().any(|placed| placed.host == *host))
        .map(|group| group.id.0.clone())
        .collect()
}

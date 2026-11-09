//! Loading this process's ranks, one thread per rank: the collective groups they agree over, each
//! rank's model, weights and resident owners ([`LoadedRank`]), and the schedulers the scheduler
//! node builds over the loaded ranks' workers.

use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use proto::config::EngineArgs;
use worker::device::Worker;
use worker::memory::Memory;
use worker::memory::retained::Retained;
use worker::memory::weights::WeightStore;
use worker::parallel::host::{HostGroup, Rendezvous, STARTUP_TIMEOUT};

use super::launch::Epoch;
use crate::rank::ModelRunner;

/// One rank of this process, loaded: what its worker is built from.
pub(super) struct LoadedRank {
    /// The rank's own model runner, whose one device is the rank's.
    pub(super) model_runner: ModelRunner,
    /// The weights its load reached, in order: what a weight update pairs against.
    pub(super) weights: Vec<worker::memory::weights::LoadedWeight>,
    /// The rank's Storage Server client, whose scope is the rank's.
    pub(super) client: Arc<Retained>,
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

/// Loads every rank of this process, each on a thread of its own.
///
/// The ranks agree their capacity and exchange their communicators' ids over one host group, whose
/// collectives every member must reach, so they run side by side rather than one after another; the
/// weights of four ranks then also load at once. The group is the one the capacity plan
/// agrees over, formed here over the loopback because this process holds all its members.
pub(super) fn load_ranks(
    model_runners: Vec<ModelRunner>,
    storage: &ipc::StorageLaunch,
    launch: &EngineArgs,
    epoch: &Epoch,
) -> Result<Vec<LoadedRank>> {
    let count = model_runners.len();
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0))
        .context("binding the rendezvous of this process's ranks")?;
    let address = listener
        .local_addr()
        .context("reading the rendezvous of this process's ranks")?
        .to_string();
    let mut listener = Some(listener);
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
                        index,
                        count,
                        storage,
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

/// Joins the group, opens the rank's collectives, loads its partition and opens its memory.
fn load_rank(
    model_runner: ModelRunner,
    rendezvous: Rendezvous,
    index: usize,
    count: usize,
    storage: &ipc::StorageLaunch,
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
        .find(|cache| cache.worker_group == launch.group && !cache.segments.is_empty())
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
    let host = HostGroup::connect(rendezvous, index, count, STARTUP_TIMEOUT)
        .context("joining this process's ranks")?;
    // The rank's Storage Server and its claim are the node's, which the daemon answered for; the
    // retained weights outlive the epoch, so the store is built once, before the model loads,
    // and keeps what it maps for the process's life.
    let client = Retained::connect(
        storage,
        &launch.group,
        model_runner.rank(),
        epoch.generation,
        epoch.predecessor_epoch,
    )
    .context("connecting to the Storage Server")?;

    // A weight's region is one device's address, so the store maps on the device this rank's model
    // is loaded for; the client is shared, the mapping is not.
    let device = *model_runner
        .devices()
        .first()
        .context("this rank serves no device to map its weights on")?;
    let shape = model_runner.parallel_shape()?;

    // NCCL keeps its own device buffers, so a communicator opens before the capacity plan measures
    // what is left of the device.
    let groups = crate::rank::RankGroups {
        sequence: group_over(&host, device, shape.sequence, shape.sequence_rank, count)
            .context("forming the sequence group")?,
        tensor: group_over(&host, device, shape.tensor, shape.tensor_rank, count)
            .context("forming the tensor group")?,
    };
    let mut weights = WeightStore::new(Arc::clone(&client), device);

    // The load and the capacity plan are one call because the order between them is the plan's: the
    // weights are mapped first, so what the plan measures is what is left of the device rather than
    // what was there before the model.
    let (model, memory, inventory, graph) = model_runner
        .load_and_open(&mut weights, &mut &*client, &host, &epoch.experts)
        .context("loading the epoch's model")?;
    tracing::info!(rank = model_runner.rank(), "the rank loaded its model");
    let weights = weights.take_log();
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

/// The collective group of `size` ranks this rank is member `rank` of, absent for a group of one.
///
/// The first member draws the group's id and the host group carries it (`parallel`'s statement of
/// what that channel is for); `Communicator::init` then blocks until every member joined.
fn group_over(
    host: &HostGroup,
    device: platform::DeviceId,
    size: usize,
    rank: usize,
    ranks: usize,
) -> Result<Option<Arc<nn::Group>>> {
    if size == 1 {
        return Ok(None);
    }
    if size != ranks {
        bail!(
            "a parallel group of {size} ranks in a process of {ranks}: this process forms one \
             group of all its ranks, and a deployment of several groups per host is not served"
        );
    }
    let id = match rank {
        0 => Some(platform::UniqueId::new()?),
        _ => None,
    };
    let bytes = host.broadcast_bytes(id.as_ref().map(|id| &id.0[..]), 0)?;
    let id = platform::UniqueId(bytes.try_into().map_err(|bytes: Vec<u8>| {
        anyhow::anyhow!("a communicator's id is 128 bytes, not {}", bytes.len())
    })?);
    let size = i32::try_from(size).context("a group wider than NCCL addresses")?;
    let member = i32::try_from(rank).context("a rank past what NCCL addresses")?;
    let communicator = platform::Communicator::init(device, size, member, &id)
        .context("joining a communicator")?;
    Ok(Some(Arc::new(nn::Group::new(communicator, size as u64)?)))
}

/// This node's scheduler threads, one per DP rank.
///
/// A node that is not the scheduler node serves only what its peers' schedulers ask of it, so it
/// builds none. The scheduler node builds one thread per data-parallel rank, each handed this
/// node's workers through `worker::Batches`: the scheduler forms a batch, the worker issues
/// it, and the scheduler reads the result - nothing crosses threads except the batch and its
/// result.
#[expect(
    clippy::too_many_arguments,
    reason = "a rank's schedulers are built from its launch, model, workers, memory and wakes"
)]
pub(super) fn schedulers(
    model_runner: &ModelRunner,
    launch: &EngineArgs,
    model: &crate::rank::Loaded,
    workers: &[Arc<Worker>],
    rank: &ipc::MainProcessInfo,
    memory: &Memory,
    control: ipc::Wake,
    data: &Arc<worker::data_plane::DataPlane>,
) -> Result<Vec<scheduler::SchedulerHandle>> {
    if !launch.scheduler_node {
        return Ok(Vec::new());
    }
    if workers.is_empty() {
        bail!("the scheduler node holds no device to serve a batch on");
    }

    // The host KV cache copies a rank's pages on that rank's copy stream, into the pool the one rank
    // of this scheduler's inventory states. A scheduler of an expert union serves one device, as a
    // scheduler of one rank does, and its peers' schedulers keep their own pools of the same pages:
    // a page holds what the union computed. A scheduler over several devices would copy one
    // device's share of every page, so it keeps none, and no text model runs one today: a rank
    // maps its model on one device.
    let host_cache = model_runner.host_kv_cache(model) && workers.len() == 1;

    // One scheduler per scheduler thread: it reads the engine configuration as this rank resolves
    // it, with the model's facts, and the limits this rank captured.
    let (args, limits) = model_runner.scheduler_args(launch, model, host_cache)?;
    let mut rank = rank.clone();
    if host_cache && let Some(cache) = rank.prefix_cache.as_mut() {
        cache.host_token_bytes = memory.host_token_bytes();
    }

    // The scheduler submits through the same devices this process drives, so a batch and a
    // switchover reach one worker thread per device and never two.
    // The scheduler plans against a description of the pool, and this process is the one pool it
    // holds: its single rank is this one, its capacities are the launch's (the rank reports what it
    // holds, not what it was launched for), and its longest sequence is the model's bound. The
    // inventory is the worker's to hold and the boot's to state, because neither number can be
    // derived from a batch.
    let inventory = worker::batch::Inventory {
        group: launch.group.clone(),
        ranks: vec![rank.clone()],
        layout: worker::batch::PoolLayout {
            queue_depth: model_runner.queue_depth(),
            ranks: model_runner.world_size(),
            // Every component the rank reports is placed on the group's ranks. A text rank reports
            // the one default component, and a media rank one per block of its pipeline, which the
            // scheduler reads a call's width and owner from.
            components: rank
                .components
                .iter()
                .flat_map(|component| {
                    scheduler::WorkerGroupConfig::single_component(
                        &component.name,
                        model_runner.world_size(),
                    )
                })
                .collect(),
        },
        max_model_len: model_runner.engine_config().model.max_model_len,
    };
    let mut stated = worker::batch::Workers::shared(workers.iter().map(Arc::clone).collect())?
        .stating(vec![inventory])
        .with_data_plane(Arc::clone(data));
    if host_cache {
        stated = stated.keeping_host_cache(worker::batch::HostCacheKeys {
            fingerprints: vec![data.serving_identity()?; memory.cache_groups()],
            endpoint: None,
        });
    }
    let batches: Arc<dyn worker::batch::Batches> = Arc::new(stated);

    // The control thread forwards what a scheduler produced, so it sleeps until a step says there
    // is something: a request's tokens reach the daemon when they exist, not at the next pass.
    let progress = scheduler::CommandWaker::new(move || control.wake());
    let handle =
        scheduler::SchedulerHandle::over_batches_telling(&args, limits, batches, progress)?;
    Ok(vec![handle])
}

//! Starting this process's ranks (`specs/engine-worker-port.md` §1.1) in two halves around the
//! session the engine opens with its host's buffer pool (`specs/buffer-pool.md` §14.1, "A session
//! opens before its ranks map memory"):
//!
//! 1. [`load_ranks`] loads each rank on a thread of its own up to the division of its GPU's
//!    quota: its Storage Server client and quota, the host groups it agrees over, its tensor
//!    group's communicator, the model's meta load and the runner's profile run, the kernels the
//!    profile run names, the weights, and the units of its cache, agreed over its host group.
//! 2. [`layout`] states the session's layout from those units; its device KV regions state no
//!    segment and ask for no unit, since none of their memory exists yet. The head opens the
//!    session with it.
//! 3. [`open_ranks`] maps on every rank the segments the answer's table states for its region,
//!    creates the segments that cover the rest of the region's units, and opens the graph pool,
//!    the cache over those segments, the recurrent state, the scratch pages and the serving
//!    runner. It returns the segments it created, which the head grants, and the schedulers
//!    then grow their regions into them by `Resize`.
//! 4. [`bind`] binds every rank to the regions the answer numbered, [`spawn`] spawns the
//!    executors over the ranks' runners, and [`capacities`] states what each worker group's
//!    ranks serve.
//!
//! One sequence serves a first epoch, a successor of the same layout and a relaunch: a region an
//! earlier session named has its table, which the ranks map, and creates nothing more.
//!
//! A start that is backed by nothing outside the process ([`Backing::Meta`]) takes the same
//! steps over owners that hold no device: its loads stop after the profile run, its "segments"
//! are statements of units with no memory behind them, and it serves generation
//! [`FIRST_GENERATION`] of no kernel. That is what a recorded start reads and what the engine's
//! host tests start.

use std::net::TcpListener;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use anyhow::{Context as _, Result, bail};
use ipc::Capacity;
use ipc::batch::Batch;
use nn::WeightKey;
use nn::family::TextModel;
use ops::Generation;
use platform::{DeviceId, Stream};
use proto::config::{EngineArgs, WorkerGroupConfig};
use storage::QueuePair;
use storage::runtime::{Class, Group, GroupId, Open, Opened, Region, Segments, Tier, VERSION};
use worker::executor::peers::Remote;
use worker::executor::{COPY_BUDGET, Executor, Executors, Sequences};
use worker::graphs::Graphs;
use worker::memory::cache::{CacheLayout, CacheRegions, KvCacheMemory, host};
use worker::memory::client::StorageClient;
use worker::memory::graph_pool::GraphPool;
use worker::memory::state::StateMemory;
use worker::memory::tables::BlockTables;
use worker::memory::weights::WeightStore;
use worker::parallel::host::{HostGroup, STARTUP_TIMEOUT};
use worker::runners::causal_lm::experts::JoinHook;
use worker::runners::causal_lm::{CausalLmConfig, CausalLmRunner};
use worker::{Runner, RunnerId};

use crate::rank::memory::{Peaks, fewest_units, granularity, plane_segments, scratch_bytes};
use crate::rank::text::{Keys, keys, profile};
use crate::rank::{Collective, ModelRunner, RankModel, Weights};

/// The number of the generation a start serves: its kernels on a device, none on the meta device.
/// A start is not an update (`specs/updates.md` §2): nothing is kept beside it.
pub(crate) const FIRST_GENERATION: u64 = 1;

/// What resolves the kernels a rank's profile run names into the generation the rank serves on
/// its device: the engine's module load at the Build Server (step 9).
pub(crate) type Resolve<'a> = dyn Fn(DeviceId, &Keys) -> Result<Arc<Generation>> + Sync + 'a;

/// The executors of the members of the node's expert union, in member order, once they are
/// spawned: what each member's join hook submits join batches to. Weak, because each member's
/// runner holds its hook and its executor holds its runner: the epoch's close owns the executors.
type Members = Arc<OnceLock<Vec<Weak<Executor>>>>;

/// The host KV cache's token count over the device KV cache's when `kv_cache_config.host_ratio`
/// is unset (SGLang's `--hicache-ratio` default).
const HOST_RATIO: f64 = 2.0;

/// The product kinds of the session's classes, one class each after the cache groups' and before
/// the class of stored trajectories, in `ipc::product` order.
const PRODUCTS: usize = 13;

/// What the ranks of a start load onto.
#[derive(Clone, Copy)]
pub(crate) enum Backing<'a> {
    /// The node's Storage Server, whose claim the daemon answered: each rank publishes its weights
    /// there, divides its GPU's quota and creates its cache's segments, and its tensor group
    /// forms its communicator. `kernels` resolves the rank's profiled keys into its serving
    /// generation.
    Serving {
        storage: &'a ipc::StorageLaunch,
        kernels: &'a Resolve<'a>,
    },
    /// Nothing outside this process: each rank's model is loaded from the shard headers alone
    /// ([`nn::WeightLoader::meta`]), its runner and owners hold no device, its tensor group
    /// states its split with no communicator, and it serves generation [`FIRST_GENERATION`] of
    /// no kernel.
    Meta,
}

/// One rank of this process: loaded to its division by [`load_ranks`], then opened over the
/// session's memory by [`open_ranks`], which is what its executor is spawned over.
pub(crate) struct LoadedRank {
    /// The rank's own model runner, whose one device is the rank's.
    pub(crate) model_runner: ModelRunner,
    /// The rank's Storage Server client, whose scope is the rank's; a meta start has none.
    pub(crate) client: Option<Arc<StorageClient>>,
    /// The store the rank's weights are retained in, which a second load publishes beside;
    /// `None` for a meta start.
    pub(crate) store: Option<Arc<WeightStore>>,
    /// The weights its load reached, in order, each with whether the model writes it after the
    /// load: what a weight update pairs against. Empty for a meta start.
    pub(crate) weights: Vec<(WeightKey, bool)>,
    /// The rank's partition of the model.
    pub(crate) model: Arc<dyn TextModel>,
    /// The units of the rank's cache, the padding unit included: the division's count, which every
    /// rank of its host group agreed, in whole segments.
    pub(crate) units: u32,
    /// What the profile run states of the rank's runner, which its pools are sized by.
    pub(crate) peaks: Peaks,
    /// The rank's tensor group's communicator; `None` for a group of one and on the meta device.
    pub(crate) tensor: Option<Arc<backend::Group>>,
    /// The config of the rank's runner: the profiled runner's until the rank is opened, the
    /// serving runner's after.
    pub(crate) config: CausalLmConfig,
    /// The rank's cache and recurrent state, which its runner reads and the data plane operates:
    /// the profile run's owners, which hold no device, until the rank is opened over the
    /// session's memory.
    pub(crate) cache: Arc<Mutex<KvCacheMemory>>,
    pub(crate) state: Option<Arc<Mutex<StateMemory>>>,
    /// The device's graph and scratch pools, which the runner's graphs address, once the rank is
    /// opened; `None` for a meta start.
    pub(crate) pool: Option<Arc<Mutex<GraphPool>>>,
    /// The rank's serving runner, from its opening until its executor takes it ([`spawn`]).
    pub(crate) runner: Option<CausalLmRunner>,
    /// Every kernel the rank launches.
    pub(crate) keys: Keys,
    /// The generation the rank serves.
    pub(crate) kernels: Arc<Generation>,
    /// The node's expert union this rank is a member of: the union's host group, the members'
    /// executors once [`spawn`] set them, and the rank's place among them; `None` outside a
    /// union.
    pub(crate) union: Option<(Arc<HostGroup>, Members, usize)>,
}

impl LoadedRank {
    /// What the data plane holds of this rank once `executor` serves it
    /// ([`crate::data_plane::Rank`]): the owners of its device's storage, a stream of the device
    /// for the data plane's own operations, and how it loads its model again; `None` for a rank of
    /// a meta start, which holds no device.
    ///
    /// # Errors
    ///
    /// When the device gives no stream.
    pub(crate) fn data_rank(
        &self,
        executor: &Arc<Executor>,
    ) -> Result<Option<crate::data_plane::Rank>> {
        let (Some(client), Some(store), Some(pool)) = (&self.client, &self.store, &self.pool)
        else {
            return Ok(None);
        };
        let device = self.device()?;

        Ok(Some(crate::data_plane::Rank {
            rank: self.model_runner.rank(),
            executor: Arc::downgrade(executor),
            device,
            client: Arc::clone(client),
            weights: Arc::clone(store),
            cache: Some(Arc::clone(&self.cache)),
            state: self.state.clone(),
            graphs: Arc::clone(pool),
            stream: Stream::new(device)
                .with_context(|| format!("creating device {}'s stream", device.0))?,
            // A communicator states no bytes NVML attributes to no process (`backend::Group`).
            unattributed: 0,
            model: Arc::new(RankModel::new(self.model_runner.clone())),
            serving: self.weights.clone(),
        }))
    }

    /// The rank's one device.
    fn device(&self) -> Result<DeviceId> {
        self.model_runner
            .devices()
            .first()
            .copied()
            .context("this rank serves no device")
    }
}

/// Where the ranks of a process's worker group form their host group.
pub(crate) enum Meeting {
    /// This process holds every rank: they meet over the loopback.
    Local,
    /// The group has ranks on other hosts: its members meet at `address`, which the process that
    /// holds rank 0 serves on `listener`, the socket its daemon reserved; a peer holds none.
    World {
        address: String,
        listener: Option<TcpListener>,
    },
}

/// Loads every rank of this process to its division, each on a thread of its own, onto
/// `backing` (steps 1-15).
///
/// The ranks agree their cache's units and exchange their communicators' ids over one host group,
/// whose collectives every member must reach, so they run side by side rather than one after
/// another; the weights of four ranks then also load at once. The group is formed over the
/// loopback when this process holds all its members, and at the group's rendezvous
/// ([`Meeting::World`]) when other hosts hold some, each rank a member under its rank number in the
/// group. The members of an expert union, which lives on one node's devices, meet once more over
/// the loopback for the union's own collectives.
///
/// # Errors
///
/// The first rank's failure, or a meeting the process's ranks do not fit.
pub(crate) fn load_ranks(
    model_runners: Vec<ModelRunner>,
    backing: Backing<'_>,
    launch: &EngineArgs,
    meeting: Meeting,
) -> Result<Vec<LoadedRank>> {
    let count = model_runners.len();
    // The members' numbers: this process's place among them, and how many there are.
    let (first, members, address, mut listener) = match meeting {
        Meeting::Local => {
            let (address, listener) = loopback("the rendezvous of this process's ranks")?;
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
    let union = model_runners
        .first()
        .and_then(ModelRunner::union)
        .map(|_| loopback("the rendezvous of this node's expert union"))
        .transpose()?;
    let (union_address, mut union_listener) = match union {
        Some((address, listener)) => (address, Some(listener)),
        None => (String::new(), None),
    };
    // One join target list for the union, which `spawn` fills with its members' executors.
    let targets: Members = Arc::default();

    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(count);
        for (index, model_runner) in model_runners.into_iter().enumerate() {
            let meeting = (address.clone(), listener.take());
            let union = model_runner.union().map(|_| {
                (
                    union_address.clone(),
                    union_listener.take(),
                    Arc::clone(&targets),
                )
            });
            let handle = std::thread::Builder::new()
                .name(format!("rank-load-{index}"))
                .spawn_scoped(scope, move || {
                    load_rank(
                        model_runner,
                        meeting,
                        (first + index, members),
                        union,
                        backing,
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

/// A listener on the loopback for the members of a host group of this process, and its address.
fn loopback(what: &str) -> Result<(String, TcpListener)> {
    let listener =
        TcpListener::bind(("127.0.0.1", 0)).with_context(|| format!("binding {what}"))?;
    let address = listener
        .local_addr()
        .with_context(|| format!("reading {what}"))?
        .to_string();
    Ok((address, listener))
}

/// Loads one rank onto `backing` to its division: it joins the process's host group at
/// `meeting` as `member` of `members` and, in an expert union, the union's host group, then runs
/// steps 1-15 of the rank's start (module documentation).
fn load_rank(
    model_runner: ModelRunner,
    meeting: (String, Option<TcpListener>),
    (member, members): (usize, usize),
    union: Option<(String, Option<TcpListener>, Members)>,
    backing: Backing<'_>,
) -> Result<LoadedRank> {
    let device = *model_runner
        .devices()
        .first()
        .context("this rank serves no device")?;

    // Step 1: the rank's Storage Server client, and the epoch's limit of its GPU.
    let client = match backing {
        Backing::Serving { storage, .. } => {
            let client = StorageClient::connect(
                storage,
                model_runner.group(),
                model_runner.rank(),
                model_runner.launch().engine,
            )
            .context("connecting to the Storage Server")?;
            client
                .state_quota(device, model_runner.engine_config().mem_fraction_static)
                .context("stating this rank's quota of its device")?;
            Some(client)
        }
        Backing::Meta => None,
    };

    // Step 2: the host group the division agrees over, and the expert union's.
    let (address, listener) = meeting;
    let host = HostGroup::connect(&address, listener, member, members, STARTUP_TIMEOUT)
        .context("joining this process's ranks")?;
    let union = match union {
        Some((address, listener, targets)) => {
            let (members, place) = model_runner
                .union()
                .context("a union member states its union")?;
            let group = HostGroup::connect(
                &address,
                listener,
                place as usize,
                members as usize,
                STARTUP_TIMEOUT,
            )
            .context("joining this node's expert union")?;
            Some((Arc::new(group), targets, place as usize))
        }
        None => None,
    };

    // Step 3: the tensor group's communicator, opened before any device measurement because
    // NCCL keeps device buffers. A text rank's tensor group is its worker group's ranks, one span
    // of the host group among the deployment's groups.
    let tensor = model_runner.tensor_spans()?;
    let tensor = match tensor {
        Some(collective) => group_over(&host, device, &collective, client.is_some())
            .context("forming the tensor group")?,
        None => None,
    };

    // Steps 4-8: the meta load, the cache's layout, the meta owners and the profiled runner.
    let experts = model_runner.experts();
    let meta = model_runner
        .load_model(Weights::Meta, &experts)
        .context("loading the epoch's model from its checkpoint's headers")?;
    let layout = Arc::new(model_runner.cache_layout(meta.as_ref())?);
    let least = fewest_units(&layout);
    let meta_config = model_runner.causal_config(meta.as_ref(), &layout, least)?;
    let meta_cache = Arc::new(Mutex::new(KvCacheMemory::meta(
        model_runner.cache_config(Arc::clone(&layout), least, None)?,
    )?));
    let state_config = model_runner.state_config(meta.as_ref(), model_runner.buckets())?;
    let meta_state = match &state_config {
        Some(config) => Some(Arc::new(Mutex::new(StateMemory::meta(
            config.clone(),
            worker::lock(&meta_cache).config(),
        )?))),
        None => None,
    };
    let meta_member = match &union {
        Some((group, targets, place)) => model_runner.union_member(
            device,
            meta.as_ref(),
            Arc::clone(group),
            join_hook(Arc::clone(targets), *place),
        )?,
        None => None,
    };
    let inputs = CausalLmRunner::input_bytes(&meta_config)?;
    let unserved = Arc::new(Generation::new(0, []));
    let profiled = model_runner.runner(
        Arc::clone(&meta),
        meta_config.clone(),
        Graphs::meta(Arc::clone(&unserved), inputs),
        None,
        &meta_cache,
        meta_state.as_ref(),
        None,
        meta_member,
        &unserved,
    )?;
    profile(&profiled)?;
    let keys = keys(
        &profiled,
        &meta_cache,
        meta_state.as_ref().map(|state| state.as_ref()),
    )?;
    let graphs = worker::Runner::graphs(&profiled);
    let peaks = Peaks {
        graph: graphs.peak(),
        shadow: graphs.shadow_peak() + scratch_bytes(&meta_config, &layout),
        inputs: graphs.input_peak(),
    };
    drop(profiled);

    let (storage, resolve) = match backing {
        Backing::Serving { storage, kernels } => (storage, kernels),
        Backing::Meta => {
            // The meta device allocates nothing, so each unit is a whole granule of it: its cache
            // is the fewest units it serves with, rounded up to at most eight segments of whole
            // units.
            let units = worker::memory::cache::segments::segmented_units_above(
                least,
                layout.plane_bytes,
                layout.plane_bytes,
            );
            return Ok(LoadedRank {
                model_runner,
                client: None,
                store: None,
                weights: Vec::new(),
                model: meta,
                units,
                peaks,
                tensor,
                config: meta_config,
                cache: meta_cache,
                state: meta_state,
                pool: None,
                runner: None,
                keys,
                kernels: Arc::new(Generation::new(FIRST_GENERATION, [])),
                union,
            });
        }
    };
    let client = client.context("a served rank holds its Storage Server client")?;

    // Step 9: the generation the rank serves, every kernel it launches resolved.
    let kernels = resolve(device, &keys).context("resolving the rank's kernels")?;

    // Step 10: the weights, retained at the Storage Server and mapped on the device.
    let store = Arc::new(WeightStore::new(Arc::clone(&client), device)?);
    let model = model_runner
        .load_model(Weights::Retained(store.as_ref()), &experts)
        .context("loading the epoch's model")?;
    let weights = store
        .finish()
        .context("publishing the granules of the epoch's packed weights")?;

    // Steps 11-15: the division of the GPU's quota into the pools, the fixed regions and the
    // cache's units.
    let resident = state_config
        .as_ref()
        .map_or(0, |config| config.bytes(&layout));
    let slot_bytes = CausalLmRunner::slot_bytes(&meta_config, model.as_ref())?
        + CausalLmRunner::kept_bytes(&meta_config, model.as_ref())?;
    let fixed = worker::lock(&meta_cache).config().fixed_bytes();
    let pools = model_runner.pools(device, &layout, peaks)?;
    let units = model_runner.divide(
        &client,
        device,
        &host,
        model.as_ref(),
        &layout,
        pools,
        resident,
        slot_bytes,
        fixed,
    )?;

    tracing::info!(
        rank = model_runner.rank(),
        group = model_runner.group(),
        units,
        storage = %storage.socket,
        "the rank loaded its model and divided its device's quota"
    );
    Ok(LoadedRank {
        model_runner,
        client: Some(client),
        store: Some(store),
        weights,
        model,
        units,
        peaks,
        tensor,
        config: meta_config,
        cache: meta_cache,
        state: meta_state,
        pool: None,
        runner: None,
        keys,
        kernels,
        union,
    })
}

/// The hook through which union member `place` asks every other member to join its step: a
/// join batch to each one's executor, of the group that executor serves, once [`spawn`] set the
/// members' executors in `targets`.
fn join_hook(targets: Members, place: usize) -> JoinHook {
    Arc::new(move |batch: Batch| {
        let members = targets.get().ok_or(worker::Error::State(
            "the expert union's executors are not spawned",
        ))?;
        for (index, executor) in members.iter().enumerate() {
            if index == place {
                continue;
            }
            let executor = executor.upgrade().ok_or(worker::Error::State(
                "an executor of the expert union is closed",
            ))?;
            let mut join = batch.clone();
            join.group = executor.group();
            executor.join(join)?;
        }
        Ok(())
    })
}

/// The communicator of the collective group of `collective` this rank is a member of, absent for
/// a group of one and for a start that `communicates` none.
///
/// A group that communicates forms its NCCL communicator: each group's first member draws the
/// group's id and the host group carries it, one broadcast per group of more than one member,
/// which every member of the host group takes part in, in the order the spans state;
/// `Communicator::init` then blocks until every member of the rank's own group joined. One that
/// does not states the split alone, which the runner's graphs take
/// ([`Graphs::set_group`]).
fn group_over(
    host: &HostGroup,
    device: DeviceId,
    collective: &Collective,
    communicates: bool,
) -> Result<Option<Arc<backend::Group>>> {
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
        return Ok(None);
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
    Ok(Some(Arc::new(backend::Group::new(communicator)?)))
}

/// The session's numbers of the deployment's worker groups `groups` and each scheduler's groups,
/// for `data_parallel` schedulers: the groups that serve requests in the order the schedulers
/// list them (each scheduler's block of consecutive serving groups,
/// [`WorkerGroupConfig::scheduler_blocks`]), then the dedicated expert groups, which no scheduler
/// lists and the session does not state.
///
/// # Errors
///
/// Serving groups that do not form `data_parallel` equal blocks, or more groups than a session
/// numbers.
pub(crate) fn group_numbers(
    groups: &[WorkerGroupConfig],
    data_parallel: usize,
) -> Result<(Vec<GroupId>, Vec<Vec<GroupId>>)> {
    let blocks = WorkerGroupConfig::scheduler_blocks(groups, data_parallel)?;
    let mut numbers: Vec<Option<GroupId>> = vec![None; groups.len()];
    let mut next: u8 = 0;
    let mut number = || -> Result<GroupId> {
        let id = GroupId(next);
        next = next
            .checked_add(1)
            .context("more worker groups than a session numbers")?;
        Ok(id)
    };

    let mut schedulers = Vec::with_capacity(blocks.len());
    for block in &blocks {
        let mut listed = Vec::with_capacity(block.len());
        for &index in block {
            let id = number()?;
            numbers[index] = Some(id);
            listed.push(id);
        }
        schedulers.push(listed);
    }
    let numbers = numbers
        .into_iter()
        .map(|id| id.map_or_else(&mut number, Ok))
        .collect::<Result<Vec<_>>>()?;
    Ok((numbers, schedulers))
}

/// The device memory the ranks of `group` address: its ranks' devices, as the deployment places
/// them. Groups of two sessions that state one memory read and write the same memory.
fn memory(group: &WorkerGroupConfig) -> String {
    group
        .ranks
        .iter()
        .map(|rank| format!("{}/{}", rank.host, rank.device))
        .collect::<Vec<_>>()
        .join(",")
}

/// The name of the device region of scheduler `scheduler`'s KV units of `layout` and `units`: two
/// epochs of one layout and one cache name one region, and any other layout another, whose memory
/// space is its own (`specs/buffer-pool.md` §14.1).
fn units_region(scheduler: usize, layout: &CacheLayout, units: u32) -> Result<String> {
    let mut digest = storage::key::Digest::new(b"uniserve.units-region");
    for group in 0..layout.groups.len() {
        let shape = host::page_shape(layout, group)?;
        digest.bytes(&shape.layout.0);
    }
    digest.numbers(&[
        layout.unit_bytes,
        layout.plane_bytes,
        u64::from(layout.columns),
        u64::from(units),
    ]);
    let digest = digest.finish();
    let short: String = digest[..8]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    Ok(format!("kv {scheduler} {short}"))
}

/// The layout the engine epoch `launch` states to open its session with its host's buffer pool
/// (`storage::runtime::Open`, `specs/buffer-pool.md` §11, §14.1), for ranks of the deployment's
/// one configuration whose caches are laid out as `cache` with `units` units, the padding unit
/// included, whose worker groups are `tensor` ranks wide, and whose request slots each take
/// `slot_bytes` of one rank's device memory. It is the one statement of a session's layout,
/// whoever builds the ranks behind it:
///
/// - **Regions.** Per scheduler, the device region of its KV units, the same units on every rank
///   and group of it. It states no segment and asks for no unit: its memory is the segments its
///   ranks map or create once the session is open, which the head grants, and its scheduler then
///   grows it into them by `Resize`. Per class of a cache group, when the deployment keeps a host
///   tier, the host region of its pages, one page a unit. Per worker group, the region of its
///   slot rows, numbered from 1, the epoch's own. A unit's bytes are summed over the tensor ranks
///   that hold a part of it.
/// - **Classes**, in the numbering the worker uses: cache group `g` is class `g`, then one class
///   per product kind in `ipc::product` order, then the class of stored trajectories. A text
///   deployment holds no product and no trajectory: those classes name no region.
/// - **Groups**, numbered for the session ([`group_numbers`]), each with its memory, its slot
///   region and per class its device regions; **per scheduler** its groups; and the
///   deployment's peers.
///
/// The epoch states no trial: the launch states none.
///
/// # Errors
///
/// Worker groups no scheduler block holds, or a layout the regions' numbering cannot state.
pub(crate) fn layout(
    launch: &EngineArgs,
    cache: &CacheLayout,
    units: u32,
    tensor: u32,
    slot_bytes: u64,
) -> Result<Open> {
    let config = &launch.engine_config;
    let groups = &config.parallel.worker_groups;
    let (numbers, schedulers) = group_numbers(groups, config.parallel.data_parallel_size.get())?;
    let slots = u32::try_from(config.max_running_requests.get())
        .context("more request slots than the protocol addresses")?;
    let tensor = u64::from(tensor.max(1));
    let kv = &config.kv_cache;

    let mut regions: Vec<Region> = Vec::new();
    let region_index = |regions: &Vec<Region>| -> Result<u8> {
        u8::try_from(regions.len()).context("more regions than a session numbers")
    };

    // The host region of each cache group's pages: the deployment's on the host, shared by its
    // replicas' ranks of one rank index.
    let host_tier = kv.host_tier && kv.prefix_caching;
    let mut host_regions: Vec<Option<u8>> = vec![None; cache.groups.len()];
    if host_tier {
        let pages_of = |page_bytes: u64, units_per_page: u32| -> u32 {
            let device_pages = u64::from(units.saturating_sub(1) / units_per_page.max(1));
            let pages = match kv.host_gib {
                Some(gib) => (gib.get() << 30) / (page_bytes * tensor).max(1),
                None => (device_pages as f64 * kv.host_ratio.unwrap_or(HOST_RATIO)) as u64,
            };
            u32::try_from(pages.max(1)).unwrap_or(u32::MAX)
        };
        for (index, group) in cache.groups.iter().enumerate() {
            // A draft page is given up first under pressure and never stored.
            if group.draft() {
                continue;
            }
            let shape = host::page_shape(cache, index)?;
            let digest: String = shape.layout.0[..8]
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect();
            let pages = pages_of(shape.page_bytes, group.units_per_page);
            let name = format!("host kv {index} {digest}");
            host_regions[index] = Some(region_index(&regions)?);
            regions.push(Region {
                space: name.clone(),
                name,
                tier: Tier::Host,
                unit_bytes: shape.page_bytes * tensor,
                // Each rank's share of a page lies in the part of its own rank index.
                part_bytes: shape.page_bytes,
                reserved: 0,
                units: pages,
                limit: pages,
                segments: Vec::new(),
            });
        }
    }

    // Per scheduler, the region of its KV units; per group, its slot rows.
    let slot_bytes = slot_bytes.max(1);
    let mut stated: Vec<Group> = Vec::new();
    for (scheduler, listed) in schedulers.iter().enumerate() {
        let name = units_region(scheduler, cache, units)?;
        let kv_region = region_index(&regions)?;
        regions.push(Region {
            space: name.clone(),
            name,
            tier: Tier::Device,
            unit_bytes: cache.unit_bytes * tensor,
            part_bytes: cache.unit_bytes,
            // Unit 0 is the padding unit a pooled table names where a row holds no page.
            reserved: 1,
            // No unit enters the region before the segments behind it exist (`Resize` after the
            // grants); the numbering covers the cache.
            units: 0,
            limit: units,
            segments: Vec::new(),
        });

        for id in listed {
            let index = numbers
                .iter()
                .position(|number| number == id)
                .context("a listed group is numbered")?;
            let group = &groups[index];
            let name = format!("slots {} {}", launch.epoch, id.0);
            let slot_region = region_index(&regions)?;
            regions.push(Region {
                space: name.clone(),
                name,
                tier: Tier::Device,
                // An accounting figure: the rows are memory of the ranks' own processes.
                unit_bytes: slot_bytes * tensor,
                part_bytes: slot_bytes,
                reserved: 1,
                units: slots,
                limit: slots + 1,
                segments: Vec::new(),
            });
            let mut per_class: Vec<Vec<u8>> = vec![vec![kv_region]; cache.groups.len()];
            per_class.extend((0..=PRODUCTS).map(|_| Vec::new()));
            stated.push(Group {
                id: *id,
                memory: memory(group),
                slots: slot_region,
                regions: per_class,
            });
        }
    }

    // Class `g` is cache group `g`; then the product kinds and the stored trajectories, which a
    // text deployment sizes per request and holds none of.
    let ranks_per_unit = u32::try_from(tensor).unwrap_or(u32::MAX);
    let mut classes: Vec<Class> = Vec::with_capacity(cache.groups.len() + PRODUCTS + 1);
    for (index, (group, host)) in cache.groups.iter().zip(&host_regions).enumerate() {
        let (tensors, encoding) = host::class_digests(cache, index, ranks_per_unit)?;
        classes.push(Class {
            units: group.units_per_page,
            host: host.map(|region| (region, 1)),
            state_in_slot: group.state(),
            tensors,
            encoding,
        });
    }
    // A text deployment keeps no product on the host, so no host copy of these classes is
    // written or compared: they state no tensors and no encoding.
    classes.extend((0..=PRODUCTS).map(|_| Class {
        units: 0,
        host: None,
        state_in_slot: false,
        tensors: [0; 32],
        encoding: [0; 32],
    }));
    let trajectory =
        u8::try_from(classes.len() - 1).context("more classes than a session numbers")?;

    Ok(Open {
        version: VERSION,
        epoch: launch.epoch,
        trial: false,
        regions,
        classes,
        trajectory,
        groups: stated,
        schedulers,
        peers: kv.peers.clone(),
    })
}

/// Opens every rank of `ranks` over the session `layout` opened, whose answer states each
/// region's segment table in `tables` (`Opened::segments`), each rank on a thread of its own
/// (steps 16-22):
///
/// - the segments of the rank's KV region: the table's, which the rank maps where their epoch
///   created them, then those its epoch creates for the rest of the region's units
///   ([`plane_segments`]);
/// - the graph pool with the scratch and resident pools, the cache over those segments, the
///   recurrent state, the scratch pages of paired forwards, and the serving runner in its region
///   of the graph pool with its shadow region beside it, profiled again so its buckets can be
///   captured.
///
/// On the meta device the owners hold no device and the segments no memory, so the cache, the
/// state and the runner are made over the region's units again, as the profile run's were.
///
/// Returns, per KV region by name, the segments its ranks created, as `(segment, first unit,
/// units)`: the head grants each once every rank that maps the region holds it.
///
/// # Errors
///
/// A rank whose group the session does not state, ranks of one region that created different
/// segments, or the first rank's failure.
pub(crate) fn open_ranks(
    ranks: &mut [LoadedRank],
    layout: &Open,
    tables: &[Segments],
) -> Result<Vec<(String, Segments)>> {
    // Each rank's KV region and its table, read before the ranks go to their threads.
    let mut regions = Vec::with_capacity(ranks.len());
    for rank in ranks.iter() {
        let (_, kv, _) = stated_regions(rank, layout)?;
        let table = tables
            .get(usize::from(kv))
            .with_context(|| format!("the answer states no segment table of region {kv}"))?;
        regions.push((layout.regions[usize::from(kv)].name.clone(), table));
    }

    let opened: Vec<Segments> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(ranks.len());
        for (index, (rank, (_, table))) in ranks.iter_mut().zip(&regions).enumerate() {
            let handle = std::thread::Builder::new()
                .name(format!("rank-open-{index}"))
                .spawn_scoped(scope, move || open_rank(rank, table))
                .context("starting a rank's opening")?;
            handles.push(handle);
        }
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("a rank's opening panicked"))?
            })
            .collect::<Result<Vec<_>>>()
    })?;

    // Every rank that maps a region creates the same segments of it, which the head grants once.
    let mut created: Vec<(String, Segments)> = Vec::new();
    for ((name, _), segments) in regions.into_iter().zip(opened) {
        match created.iter().find(|(region, _)| *region == name) {
            Some((_, first)) if *first != segments => bail!(
                "the ranks of region {name} created different segments: {first:?} and \
                 {segments:?}"
            ),
            Some(_) => {}
            None => created.push((name, segments)),
        }
    }
    Ok(created)
}

/// Opens one rank over its KV region's segment table `table` (steps 16-22, [`open_ranks`]).
/// Returns the segments it created, as `(segment, first unit, units)`.
fn open_rank(rank: &mut LoadedRank, table: &Segments) -> Result<Segments> {
    let device = rank.device()?;
    let model_runner = &rank.model_runner;
    let layout = Arc::clone(worker::lock(&rank.cache).layout());
    let units = rank.units;
    let state_config = model_runner.state_config(rank.model.as_ref(), model_runner.buckets())?;
    let config = model_runner.causal_config(rank.model.as_ref(), &layout, units)?;
    let inputs = CausalLmRunner::input_bytes(&config)?;
    let member = match &rank.union {
        Some((group, targets, place)) => model_runner.union_member(
            device,
            rank.model.as_ref(),
            Arc::clone(group),
            join_hook(Arc::clone(targets), *place),
        )?,
        None => None,
    };

    // Step 16: the segments of the region the table states, mapped, and the rest, created. A
    // segment's every part is whole granules of the device's allocations; the meta device
    // allocates nothing, and each unit is a whole granule of it.
    let granularity = match &rank.client {
        Some(_) => granularity(device)?,
        None => layout.plane_bytes,
    };
    let (segments, created) = plane_segments(
        model_runner.launch().epoch,
        units,
        &layout,
        granularity,
        table,
    )?;
    let cache_config = model_runner.cache_config(Arc::clone(&layout), units, Some(segments))?;

    let Some(client) = rank.client.clone() else {
        // Steps 18, 19 and 21-22 over owners that hold no device.
        let cache = Arc::new(Mutex::new(KvCacheMemory::meta(cache_config)?));
        let state = match state_config {
            Some(state) => Some(Arc::new(Mutex::new(StateMemory::meta(
                state,
                worker::lock(&cache).config(),
            )?))),
            None => None,
        };
        let runner = model_runner.runner(
            Arc::clone(&rank.model),
            config.clone(),
            Graphs::meta(Arc::clone(&rank.kernels), inputs),
            None,
            &cache,
            state.as_ref(),
            None,
            member,
            &rank.kernels,
        )?;
        profile(&runner)?;

        rank.config = config;
        rank.cache = cache;
        rank.state = state;
        rank.runner = Some(runner);
        return Ok(created);
    };

    // Step 17: the graph pool, with the scratch and resident pools it reserves.
    let (_, scratch) = model_runner.pools(device, &layout, rank.peaks)?;
    let resident = state_config
        .as_ref()
        .map_or(0, |state| state.bytes(&layout));
    let mut pool = GraphPool::open(
        Arc::clone(&client),
        device,
        model_runner.engine_config().mem_fraction_static,
        &[(RunnerId(0), rank.peaks.graph)],
        scratch,
        resident,
    )?;

    // Step 18: the cache, whose segments the rank maps or creates at the Storage Server.
    let stream =
        Stream::new(device).with_context(|| format!("creating device {}'s stream", device.0))?;
    let mut cache = KvCacheMemory::open(cache_config, device, &*client, &stream)
        .context("opening the rank's KV cache")?;

    // Step 19: the recurrent state, from the resident pool.
    let state = match state_config {
        Some(state) => Some(Arc::new(Mutex::new(StateMemory::open(
            state,
            cache.config(),
            pool.allocator(),
            stream.clone(),
            Arc::clone(&rank.kernels),
        )?))),
        None => None,
    };

    // Step 20: the scratch pages of paired forwards, from the scratch pool.
    let scratch_bytes = scratch_bytes(&config, &layout);
    if config.scratch_pages > 0 {
        let pages = cache.scratch_pages_within(scratch_bytes);
        cache.open_scratch(pages, Some((pool.allocator(), &stream)))?;
    }
    let cache = Arc::new(Mutex::new(cache));

    // Steps 21-22: the serving runner in its region of the graph pool, its shadow region beside
    // it (the profile run's shadow figure less the scratch pages, which the scratch pool holds),
    // profiled again so its buckets can be captured.
    let graphs = Graphs::new(
        stream.clone(),
        Arc::clone(&rank.kernels),
        pool.region(RunnerId(0))?,
        inputs,
        rank.peaks.inputs,
        2,
    )?;
    let runner = model_runner.runner(
        Arc::clone(&rank.model),
        config.clone(),
        graphs,
        Some(stream),
        &cache,
        state.as_ref(),
        rank.tensor.clone(),
        member,
        &rank.kernels,
    )?;
    let shadow = rank.peaks.shadow.saturating_sub(scratch_bytes);
    worker::Runner::graphs(&runner).set_shadow(pool.shadow(shadow)?)?;
    profile(&runner)?;

    tracing::info!(
        rank = model_runner.rank(),
        group = model_runner.group(),
        units,
        created = created.len(),
        "the rank opened its memory over the session's region"
    );
    rank.config = config;
    rank.cache = cache;
    rank.state = state;
    rank.pool = Some(Arc::new(Mutex::new(pool)));
    rank.runner = Some(runner);
    Ok(created)
}

/// Binds each rank of `ranks` to the session `layout` opened and `opened` answered: a served rank
/// states where its part of its scheduler's KV units lies in its group's memory (`Place`), and
/// every rank's cache is told the numbers of the regions its copies name
/// (`KvCacheMemory::set_regions`).
///
/// # Errors
///
/// A rank whose group the session does not state, or the Storage Server's refusal.
pub(crate) fn bind(ranks: &[LoadedRank], layout: &Open, opened: &Opened) -> Result<()> {
    for rank in ranks {
        let (group, kv, host) = stated_regions(rank, layout)?;
        let number = |index: u8| -> Result<u8> {
            opened
                .regions
                .get(usize::from(index))
                .map(|(number, _)| *number)
                .with_context(|| format!("the answer numbers no region {index}"))
        };

        let mut cache = worker::lock(&rank.cache);
        if let Some(client) = &rank.client {
            crate::storage_client::place(
                client,
                &layout.regions[usize::from(kv)].name,
                &group.memory,
                &cache.info().plane_runs,
            )?;
        }

        let groups = cache.layout().groups.len();
        let host = host
            .into_iter()
            .map(|region| region.map(number).transpose())
            .collect::<Result<Vec<_>>>()?;
        cache.set_regions(CacheRegions {
            units: number(kv)?,
            host,
            sources: vec![None; groups],
        })?;
    }
    Ok(())
}

/// The session's statement of `rank`'s group, the index of its KV region, and per cache group the
/// index of its host region.
fn stated_regions<'a>(
    rank: &LoadedRank,
    layout: &'a Open,
) -> Result<(&'a Group, u8, Vec<Option<u8>>)> {
    let id = session_group(rank, layout)?;
    let group = layout
        .groups
        .iter()
        .find(|group| group.id == id)
        .with_context(|| format!("the session states no group {}", id.0))?;
    let kv = group
        .regions
        .first()
        .and_then(|regions| regions.first())
        .copied()
        .context("the session states no KV region of the rank's group")?;
    let groups = worker::lock(&rank.cache).layout().groups.len();
    let host = layout
        .classes
        .iter()
        .take(groups)
        .map(|class| class.host.map(|(region, _)| region))
        .collect();
    Ok((group, kv, host))
}

/// The session's number of `rank`'s worker group.
fn session_group(rank: &LoadedRank, layout: &Open) -> Result<GroupId> {
    let config = &rank.model_runner.engine_config().parallel;
    let (numbers, _) = group_numbers(&config.worker_groups, config.data_parallel_size.get())?;
    let index = config
        .worker_groups
        .iter()
        .position(|group| group.id.0 == rank.model_runner.group())
        .with_context(|| {
            format!(
                "the deployment places no group {}",
                rank.model_runner.group()
            )
        })?;
    let id = numbers[index];
    if !layout.groups.iter().any(|group| group.id == id)
        && config.worker_groups[index].role.serves_requests()
    {
        bail!("the session states no group {}", id.0);
    }
    Ok(id)
}

/// Spawns one executor per rank of `ranks` over its runner (step 23), each a rank of its worker
/// group with the group's device block tables over the session's regions and the copy budget,
/// stating its ended sequence to the group's `completed` word in `pairs`, the session's queue
/// pairs (`client` maps them again); sets the expert union's join targets (step 24); and routes
/// the node's executors and `remotes`, the peer links with the groups each holds ranks of, by
/// worker group (step 27).
///
/// A group's ranks report to its sequences in the order `Executors` states: this process's
/// executors of the group, then its peer links. The unit sizes the tables count a copy by are
/// every region's of the session, so every rank of a group starts the same copies with a batch.
///
/// # Errors
///
/// A session not open, a rank whose runner was taken, or an executor that cannot start.
pub(crate) fn spawn(
    ranks: &mut [LoadedRank],
    client: &crate::storage_client::StorageClient,
    pairs: &[QueuePair],
    remotes: Vec<(Vec<GroupId>, Arc<Remote>)>,
) -> Result<(Arc<Executors>, Vec<Arc<Executor>>)> {
    let (layout, opened) = client
        .session()
        .context("the session with the buffer pool is not open")?;
    let unit_bytes: Vec<(u8, u64)> = opened
        .regions
        .iter()
        .zip(&layout.regions)
        .map(|((number, _), region)| (*number, region.unit_bytes))
        .collect();

    // Each rank's group, and its place among the group's ranks of this process.
    let mut placed: Vec<(GroupId, usize)> = Vec::with_capacity(ranks.len());
    for rank in ranks.iter() {
        let id = session_group(rank, layout)?;
        let place = placed.iter().filter(|(group, _)| *group == id).count();
        placed.push((id, place));
    }

    // One sequences per group: this process's ranks of it and the peer links that hold others,
    // writing the group's word in the registers of the scheduler that lists it.
    let mut sequences: Vec<Arc<Sequences>> = Vec::new();
    for (id, _) in &placed {
        if sequences.iter().any(|group| group.group() == *id) {
            continue;
        }
        let local = placed.iter().filter(|(group, _)| group == id).count();
        let links = remotes.iter().filter(|(held, _)| held.contains(id)).count();
        let registers = if layout.groups.iter().any(|group| group.id == *id) {
            client.completed_registers(*id, pairs)?
        } else {
            Vec::new()
        };
        sequences.push(Sequences::new(*id, local + links, registers));
    }

    let mut executors = Vec::with_capacity(ranks.len());
    for (rank, (id, place)) in ranks.iter_mut().zip(&placed) {
        let group = layout.groups.iter().find(|group| group.id == *id);
        let mapped: Vec<u8> = match group {
            Some(group) => std::iter::once(group.slots)
                .chain(group.regions.iter().flatten().copied())
                .chain(
                    layout
                        .classes
                        .iter()
                        .filter_map(|class| class.host.map(|(region, _)| region)),
                )
                .filter_map(|index| {
                    opened
                        .regions
                        .get(usize::from(index))
                        .map(|(number, _)| *number)
                })
                .collect(),
            None => Vec::new(),
        };
        let tables = BlockTables::new(*id, &mapped, &unit_bytes);
        let runner = rank
            .runner
            .take()
            .context("the rank's runner was spawned already, or the rank was never opened")?;
        let device = rank.pool.as_ref().map(|_| rank.model_runner.devices()[0]);
        let group_sequences = sequences
            .iter()
            .find(|group| group.group() == *id)
            .context("every placed group has sequences")?;
        let executor = Executor::spawn(
            device,
            vec![Box::new(runner) as Box<dyn Runner>],
            Arc::clone(&rank.kernels),
            tables,
            Arc::clone(group_sequences),
            *place,
            COPY_BUDGET,
        )?;
        executors.push(Arc::new(executor));
    }

    // The union's members ask each other to join through their executors.
    if let Some((_, targets, _)) = ranks.first().and_then(|rank| rank.union.as_ref()) {
        targets
            .set(executors.iter().map(Arc::downgrade).collect())
            .map_err(|_| anyhow::anyhow!("the expert union's executors were set already"))?;
    }

    let route = Executors::new(sequences, executors.clone(), remotes)?;
    Ok((Arc::new(route), executors))
}

/// What each worker group of `ranks` serves (`Capacity`), by the session's number of the group:
/// its first rank's statement, which every rank of a group states alike.
///
/// A cache group's page seed is `storage::key::page_seed` of its page identity
/// (`KvCacheMemory::page_identities`) for the identity of the weights this process's ranks
/// serve, in rank order (`crate::data_plane::identity_of`): what the data plane keys the ranks'
/// pages by, so a block computed by other weights or in another layout is another block.
///
/// # Errors
///
/// A rank whose group the session does not state, or a capacity its config cannot state.
pub(crate) fn capacities(ranks: &[LoadedRank], layout: &Open) -> Result<Vec<(GroupId, Capacity)>> {
    let order: Vec<(WeightKey, bool)> = ranks
        .iter()
        .flat_map(|rank| rank.weights.iter().cloned())
        .collect();
    let identity = crate::data_plane::identity_of(&order);

    let mut stated: Vec<(GroupId, Capacity)> = Vec::new();
    for rank in ranks {
        let id = session_group(rank, layout)?;
        if stated.iter().any(|(group, _)| *group == id) {
            continue;
        }
        let cache = worker::lock(&rank.cache);
        let seeds = cache
            .page_identities(&identity)
            .iter()
            .map(|identity| storage::key::page_seed(identity))
            .collect();
        let capacity = rank
            .model_runner
            .capacity(&rank.config, cache.layout(), seeds)?;
        stated.push((id, capacity));
    }
    Ok(stated)
}

#[cfg(test)]
pub(crate) mod tests;

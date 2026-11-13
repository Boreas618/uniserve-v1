//! Starting this process's ranks (`specs/engine-worker-port.md` §1.1) in two halves around the
//! session the engine opens with its host's buffer pool (`specs/buffer-pool.md` §14.1, "A session
//! opens before its ranks map memory"):
//!
//! 1. [`load_ranks`] loads each rank on a thread of its own up to the division of its GPU's
//!    quota: its Storage Server client and quota, the host groups it agrees over, its tensor
//!    group's communicator, the model's meta load and the runner's profile run, the kernels the
//!    profile run names, the weights, and its request slots and the units of its cache, agreed
//!    over its host group.
//! 2. [`layout`] states the session's layout from those units; its device KV regions state no
//!    segment and ask for no unit, since none of their memory exists yet. The head opens the
//!    session with it, and the answer numbers each region as the buffer pool holds it: a region
//!    an earlier session named keeps that session's numbering, which the ranks then take.
//! 3. [`open_ranks`] maps on every rank the segments the answer's table states for its region,
//!    creates the segments that cover the rest of the region's units as far as the GPU's quota
//!    held them at the division (the rest a grow creates), and opens the graph pool,
//!    the cache over those segments, the recurrent state, the scratch pages and the serving
//!    runner. It returns the segments it created, which the head grants, and the schedulers
//!    then grow their regions into them by `Resize`.
//! 4. [`bind`] binds every rank to the regions the answer numbered, [`spawn`] spawns the
//!    workers over the ranks' runners, and [`capacities`] states what each worker group's
//!    ranks serve.
//!
//! A rank that encodes images (`ModelRunner::encodes_images`) also profiles its vision runner at
//! its load, and its division sizes the runner's region, its region of encoder outputs and its
//! causal runner's feature buffer; the layout states a region of encoder outputs per group that
//! encodes and the host region of them, and the rank's opening takes its media memory over that
//! region from the resident pool, maps the one part of the host region its process copies with,
//! lends both runners that memory and builds the vision runner as `RunnerId(1)` beside the causal
//! runner (`specs/engine-worker-port.md` §1.1, §2.2).
//!
//! One sequence serves a first epoch, a successor of the same layout and a relaunch: a region an
//! earlier session named has its table, which the ranks map, and creates nothing more.
//!
//! A rank of a media deployment ([`Loaded::Video`], `crate::process::ranks::rank::video`) takes the
//! same steps and keeps no cache: its load profiles the media runners of the blocks its group
//! places on it and forms its text encoder's and denoiser's groups, and once every rank is loaded
//! each group's request slots are decided ([`fit_media_slots`]); the layout states every rank's
//! product buffer, the slot rows of each group with a trajectory's row, the host regions of encoder
//! outputs and stored trajectories, and the product classes with the regions each group's blocks
//! write and read; its opening takes its media memory from its graph pool's resident pool, the
//! process's first rank maps the one part of each host region, the process's memories are
//! connected, and the serving runners are built over them. It creates no segment. A device that
//! holds ranks of several groups opens one graph pool and one media memory per rank.
//!
//! A start that is backed by nothing outside the process ([`Backing::Meta`]) takes the same
//! steps over owners that hold no device: its loads stop after the profile run, its "segments"
//! are statements of units with no memory behind them, and it serves generation
//! [`FIRST_GENERATION`] of no kernel. That is what a recorded start reads and what the engine's
//! host tests start.

pub(crate) mod rank;

use std::collections::BTreeMap;
use std::net::TcpListener;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use crate::batch::Capacity;
use crate::batch::product;
use crate::batch::{Batch, Phase};
use crate::executor::Route;
use crate::executor::peers::Remote;
use crate::memory::buffers::{BufferMemory, UNIT_BYTES};
use crate::memory::cache::{CacheLayout, CacheRegions, KvCacheMemory, host};
use crate::memory::client::{StorageClient, unit_runs};
use crate::memory::division;
use crate::memory::encoders;
use crate::memory::graph_pool::GraphPool;
use crate::memory::latent;
use crate::memory::state::StateMemory;
use crate::memory::tables::BlockTables;
use crate::memory::weights::WeightStore;
use crate::parallel::host::{HostGroup, STARTUP_TIMEOUT};
use crate::runner::causal_lm::experts::JoinHook;
use crate::runner::causal_lm::{CausalLmConfig, CausalLmRunner};
use crate::runner::graphs::Graphs;
use crate::runner::thread_pool::ThreadPool;
use crate::runner::vision::VisionRunner;
use crate::runner::{Runner, RunnerId};
use crate::worker::{COPY_BUDGET, Sequences, Worker};
use anyhow::{Context as _, Result, bail};
use nn::WeightKey;
use nn::family::TextModel;
use ops::Generation;
use platform::{DeviceId, Stream};
use proto::config::{EngineArgs, WorkerGroupConfig};
use storage::QueuePair;
use storage::protocol::UnitRun;
use storage::runtime::{Class, Group, GroupId, Open, Opened, Region, Segments, Tier, VERSION};

use self::rank::{Collective, ModelRunner, RankModel, Weights, media_tasks};
use crate::process::binding::{self, Served};
use crate::process::ranks::rank::memory::{
    Peaks, fewest_units, granularity, plane_segments, scratch_bytes,
};
use crate::process::ranks::rank::text::{
    Keys, encoder_class, encodes, feature_tensors, keys, profile, vision_runner,
};
use crate::process::ranks::rank::video::{
    VideoRank, backed_kinds, boxed, group_phases, latent_pages, product_classes, product_kind,
    slot_pages,
};

/// The number of the generation a start serves: its kernels on a device, none on the meta device.
/// A start is not an update (`specs/updates.md` §2): nothing is kept beside it.
pub(crate) const FIRST_GENERATION: u64 = 1;

/// What resolves the kernels a rank's profile run names into the generation the rank serves on
/// its device: the engine's module load at the Build Server (step 9).
pub(crate) type Resolve<'a> = dyn Fn(DeviceId, &Keys) -> Result<Arc<Generation>> + Sync + 'a;

/// The workers of the members of the node's expert union, in member order, once they are
/// spawned: what each member's join hook submits join batches to. Weak, because each member's
/// runner holds its hook and its worker holds its runner: the epoch's close owns the workers.
type Members = Arc<OnceLock<Vec<Weak<Worker>>>>;

/// The host KV cache's token count over the device KV cache's when `kv_cache_config.host_ratio`
/// is unset (SGLang's `--hicache-ratio` default).
const HOST_RATIO: f64 = 2.0;

/// What the layout states of a sample a trajectory holds on a rank of a denoising group: the
/// whole video and audio latents on every rank, since a step gathers the velocity to every rank
/// of its sequence group before the solver's update. Two ranks of one statement lay every
/// trajectory out alike (`crate::memory::latent::host::share`).
const WHOLE_SAMPLES: &str = "video and audio latents, whole on every rank";

/// What the ranks of a start load onto.
#[derive(Clone, Copy)]
pub(crate) enum Backing<'a> {
    /// The node's Storage Server, whose claim the daemon answered: each rank publishes its weights
    /// there, divides its GPU's quota and creates its cache's segments, and its tensor group
    /// forms its communicator. `kernels` resolves the rank's profiled keys into its serving
    /// generation.
    Serving {
        storage: &'a crate::memory::client::StorageLaunch,
        kernels: &'a Resolve<'a>,
    },
    /// Nothing outside this process: each rank's model is loaded from the shard headers alone
    /// ([`nn::WeightLoader::meta`]), its runner and owners hold no device, its tensor group
    /// states its split with no communicator, and it serves generation [`FIRST_GENERATION`] of
    /// no kernel.
    Meta,
}

/// One rank of this process: loaded to its division by [`load_ranks`], then opened over the
/// session's memory by [`open_ranks`], which is what its worker is spawned over.
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
    /// The device's graph and scratch pools, which the runners' graphs address, once the rank is
    /// opened; `None` for a meta start.
    pub(crate) pool: Option<Arc<Mutex<GraphPool>>>,
    /// The streams the rank's runners compute on (`device`) and its copies run on (`copy`),
    /// named as the always-on counters record their device time
    /// (`crate::profiling::counters::Counters::start`); empty for a meta start.
    pub(crate) streams: Vec<(&'static str, Stream)>,
    /// Every kernel the rank launches.
    pub(crate) keys: Keys,
    /// The generation the rank serves.
    pub(crate) kernels: Arc<Generation>,
    /// What the rank computes with.
    pub(crate) loaded: Loaded,
}

/// What a rank computes with, by the kind of model its deployment serves.
pub(crate) enum Loaded {
    /// A rank of a text deployment: its causal runner over its cache.
    Text(Box<TextRank>),
    /// A rank of a media deployment: its media runners over its media memory.
    Video(Box<VideoRank>),
}

/// A rank of a text deployment as its start holds it.
pub(crate) struct TextRank {
    /// The rank's partition of the model.
    pub(crate) model: Arc<dyn TextModel>,
    /// The units of the rank's cache, the padding unit included: the division's count, which every
    /// rank of its host group agreed, in whole segments, numbering the region as if the epoch
    /// served the GPU alone; from the rank's opening on, the region's numbering as the session
    /// holds it, which an earlier epoch of the layout may have stated ([`open_ranks`]).
    pub(crate) units: u32,
    /// The units of `units` the rank's start creates segments for, which every rank of its host
    /// group agreed: the whole segments the GPU's quota held at the division. The rest are
    /// created by a grow as the epoch the rank's succeeds gives its memory back.
    pub(crate) first: u32,
    /// What the profile run states of the rank's runner, which its pools are sized by.
    pub(crate) peaks: Peaks,
    /// What the profile run states of the rank's vision runner (`RunnerId(1)`), on a rank that
    /// encodes images (`ModelRunner::encodes_images`); `None` on any other. It computes no paired
    /// forward, so it states no shadow region.
    pub(crate) vision: Option<Peaks>,
    /// The rank's tensor group's communicator; `None` for a group of one and on the meta device.
    pub(crate) tensor: Option<Arc<ops::Group>>,
    /// The config of the rank's runner: the profiled runner's until the rank is opened, the
    /// serving runner's after.
    pub(crate) config: CausalLmConfig,
    /// The rank's cache and recurrent state, which its runner reads and the weight binding
    /// operates: the profile run's owners, which hold no device, until the rank is opened over the
    /// session's memory.
    pub(crate) cache: Arc<Mutex<KvCacheMemory>>,
    pub(crate) state: Option<Arc<Mutex<StateMemory>>>,
    /// The rank's serving runner, from its opening until its worker takes it ([`spawn`]).
    pub(crate) runner: Option<CausalLmRunner>,
    /// The serving vision runner of a rank that encodes images, from its opening until its worker
    /// takes it beside the causal runner.
    pub(crate) vision_runner: Option<VisionRunner>,
    /// The node's expert union this rank is a member of: the union's host group, the members'
    /// workers once [`spawn`] set them, and the rank's place among them; `None` outside a
    /// union.
    pub(crate) union: Option<(Arc<HostGroup>, Members, usize)>,
}

impl LoadedRank {
    /// What the weight binding holds of this rank once `worker` serves it
    /// ([`crate::process::binding::Rank`]): the owners of its device's storage, a stream of the
    /// device for the weight binding's own operations, and how it loads its model again; `None`
    /// for a rank of a meta start, which holds no device.
    ///
    /// # Errors
    ///
    /// When the device gives no stream.
    pub(crate) fn data_rank(
        &self,
        worker: &Arc<Worker>,
    ) -> Result<Option<crate::process::binding::Rank>> {
        let (Some(client), Some(store), Some(pool)) = (&self.client, &self.store, &self.pool)
        else {
            return Ok(None);
        };
        let device = self.device()?;
        let (cache, state) = match &self.loaded {
            Loaded::Text(text) => (Some(Arc::clone(&text.cache)), text.state.clone()),
            Loaded::Video(_) => (None, None),
        };

        let (_, fraction) = self.model_runner.fractions()?;
        Ok(Some(crate::process::binding::Rank {
            rank: self.model_runner.rank(),
            worker: Arc::downgrade(worker),
            device,
            fraction,
            client: Arc::clone(client),
            weights: Arc::clone(store),
            cache,
            state,
            graphs: Arc::clone(pool),
            stream: Stream::new(device)
                .with_context(|| format!("creating device {}'s stream", device.0))?,
            // A communicator states no bytes NVML attributes to no process (`ops::Group`).
            unattributed: 0,
            model: Arc::new(RankModel::new(self.model_runner.clone())),
            serving: self.weights.clone(),
            encodes: self.encodes(),
        }))
    }

    /// The bytes this media rank's share of its GPU leaves for its request slots once its weights
    /// are mapped (`ModelRunner::media_left`); `None` for a rank backed by nothing, which holds
    /// no share of a device.
    ///
    /// # Errors
    ///
    /// A text rank; the Storage Server's or the driver's refusal, or a pool the quota cannot
    /// hold.
    pub(crate) fn media_left(&self) -> Result<Option<u64>> {
        let Some(client) = &self.client else {
            return Ok(None);
        };
        let video = self.video()?;
        self.model_runner
            .media_left(client, self.device()?, &video.peaks, video.buffers)
            .map(Some)
    }

    /// Whether the rank computes encoder outputs: it serves a text encoding, a chat image's
    /// vision encoding or a condition latent's encoding, whose outputs the encoder identity of
    /// its group keys (`Capacity::encoder_identity`). A text rank encodes chat images when it
    /// holds a vision runner.
    pub(crate) fn encodes(&self) -> bool {
        let serves = |phase: Phase| match &self.loaded {
            Loaded::Text(text) => {
                let images = phase == Phase::VisionEncode && text.vision.is_some();
                images || text.config.phases.contains(&phase)
            }
            Loaded::Video(video) => video.capacity.serves(phase),
        };
        [Phase::TextEncode, Phase::VisionEncode, Phase::LatentEncode]
            .into_iter()
            .any(serves)
    }

    /// The rank's one device.
    fn device(&self) -> Result<DeviceId> {
        self.model_runner
            .devices()
            .first()
            .copied()
            .context("this rank serves no device")
    }

    /// The rank of a text deployment this rank is.
    ///
    /// # Errors
    ///
    /// A rank of a media deployment.
    pub(crate) fn text(&self) -> Result<&TextRank> {
        match &self.loaded {
            Loaded::Text(text) => Ok(text.as_ref()),
            Loaded::Video(_) => bail!("rank {} is a media rank", self.model_runner.rank()),
        }
    }

    /// The rank of a media deployment this rank is.
    ///
    /// # Errors
    ///
    /// A rank of a text deployment.
    pub(crate) fn video(&self) -> Result<&VideoRank> {
        match &self.loaded {
            Loaded::Video(video) => Ok(video.as_ref()),
            Loaded::Text(_) => bail!("rank {} is a text rank", self.model_runner.rank()),
        }
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
    // One join target list for the union, which `spawn` fills with its members' workers.
    let targets: Members = Arc::default();

    let mut ranks = std::thread::scope(|scope| {
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
            .collect::<Result<Vec<LoadedRank>>>()
    })?;

    // A media group's request slots are its ranks' fewest, so they are decided once every rank
    // of the process, which holds every rank of a media group, is loaded.
    if ranks.iter().any(|rank| rank.video().is_ok()) {
        fit_media_slots(&mut ranks, LoadedRank::media_left)?;
    }
    Ok(ranks)
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
/// steps 1-15 of the rank's start (module documentation); a rank of a media deployment runs
/// [`load_video_rank`] once it joined the host group.
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

    // Step 1: the rank's Storage Server client, and the epoch's limit of its GPU: the shares of
    // every rank the deployment places on the GPU, which each of them states alike.
    let client = match backing {
        Backing::Serving { storage, .. } => {
            let client = StorageClient::connect_rank(
                storage,
                model_runner.group(),
                model_runner.rank(),
                model_runner.launch().engine,
            )
            .context("connecting to the Storage Server")?;
            let (_, fraction) = model_runner.fractions()?;
            client
                .state_quota(device, fraction)
                .context("stating this rank's quota of its device")?;
            Some(client)
        }
        Backing::Meta => None,
    };

    // Step 2: the host group the division agrees over, and the expert union's.
    let (address, listener) = meeting;
    let host = HostGroup::connect(&address, listener, member, members, STARTUP_TIMEOUT)
        .context("joining this process's ranks")?;
    if model_runner.is_video() {
        return load_video_rank(model_runner, device, client, &host, backing);
    }
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
        .load_text(Weights::Meta, &experts)
        .context("loading the epoch's model from its checkpoint's headers")?;
    let layout = Arc::new(model_runner.cache_layout(meta.as_ref())?);
    let least = fewest_units(&layout);
    let (meta_config, meta_cache, meta_state) = meta_owners(&model_runner, meta.as_ref(), &layout)?;
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
    let mut profiled = model_runner.runner(
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

    // A rank that encodes images profiles its causal runner lent a feature buffer, whose
    // prefills place image rows, and its vision runner, over a media memory of no device: the
    // region numbers it states are none of a forward's concern.
    let class = encoder_class(&layout)?;
    let images = model_runner.image_rows(meta.as_ref());
    let media = if model_runner.encodes_images(meta.as_ref())? {
        let slots = (0, model_runner.slots());
        let memory = Arc::new(BufferMemory::meta(
            model_runner.rank(),
            (1, 1),
            slots,
            0,
            &[],
        )?);
        profiled.set_features(Arc::clone(&memory), class)?;
        Some(memory)
    } else {
        None
    };
    profile(&profiled)?;
    let mut keys = keys(
        &profiled,
        &meta_cache,
        meta_state.as_ref().map(|state| state.as_ref()),
    )?;
    let graphs = crate::runner::Runner::graphs(&profiled);
    let peaks = Peaks {
        graph: graphs.peak(),
        shadow: graphs.shadow_peak(CausalLmRunner::cache_writes(&meta_config, &layout))
            + scratch_bytes(&meta_config, service::lock(&meta_cache).config()),
        inputs: graphs.input_peak(),
    };
    drop(profiled);

    let vision = match media {
        Some(memory) => {
            let inputs = VisionRunner::input_bytes(meta.as_ref(), images)?;
            let graphs = Graphs::meta(Arc::clone(&unserved), inputs);
            let model = Arc::clone(&meta);
            let runner = vision_runner(model, graphs, None, memory, (class, images))?;
            let graphs = crate::runner::Runner::graphs(&runner);
            keys.extend(graphs.keys());
            Some(Peaks {
                graph: graphs.peak(),
                shadow: 0,
                inputs: graphs.input_peak(),
            })
        }
        None => None,
    };
    tracing::info!(
        rank = model_runner.rank(),
        graph = peaks.graph,
        shadow = peaks.shadow,
        inputs = peaks.inputs,
        vision_graph = vision.map_or(0, |vision| vision.graph),
        vision_inputs = vision.map_or(0, |vision| vision.inputs),
        image_rows = images,
        "the rank's runners were profiled: the bytes of their regions, shadow regions and inputs"
    );

    let (storage, resolve) = match backing {
        Backing::Serving { storage, kernels } => (storage, kernels),
        Backing::Meta => {
            // The meta device allocates nothing, so each unit is a whole granule of it: its cache
            // is the fewest units it serves with, rounded up to at most eight segments of whole
            // units.
            let units = crate::memory::cache::segments::segmented_units_above(
                least,
                layout.plane_bytes,
                layout.plane_bytes,
            );
            return Ok(LoadedRank {
                model_runner,
                client: None,
                store: None,
                weights: Vec::new(),
                pool: None,
                streams: Vec::new(),
                keys,
                kernels: Arc::new(Generation::new(FIRST_GENERATION, [])),
                loaded: Loaded::Text(Box::new(TextRank {
                    model: meta,
                    units,
                    first: units,
                    peaks,
                    vision,
                    tensor,
                    config: meta_config,
                    cache: meta_cache,
                    state: meta_state,
                    runner: None,
                    vision_runner: None,
                    union,
                })),
            });
        }
    };
    let client = client.context("a served rank holds its Storage Server client")?;

    // Step 9: the generation the rank serves, every kernel it launches resolved.
    let kernels = resolve(device, &keys).context("resolving the rank's kernels")?;

    // Step 10: the weights, retained at the Storage Server and mapped on the device.
    let store = Arc::new(WeightStore::new(Arc::clone(&client), device)?);
    let model = model_runner
        .load_text(Weights::Retained(store.as_ref()), &experts)
        .context("loading the epoch's model")?;
    let weights = store
        .finish()
        .context("publishing the granules of the epoch's packed weights")?;

    // Steps 11-15: the division of the GPU's quota into the pools, the request slots and the
    // cache's units; on a rank that encodes images, also its vision runner's region, its region
    // of encoder outputs and its causal runner's feature buffer.
    let (graph, scratch) = model_runner.pools(device, &layout, &runner_peaks(peaks, vision))?;
    let mut regions = vec![("graph pool", graph), ("scratch pool", scratch)];
    if vision.is_some() {
        regions.push(("buffer pool", encoder_bytes(model.as_ref(), images)?));
        let features = CausalLmRunner::image_bytes(&meta_config, model.as_ref())?;
        regions.push(("image features", features));
    }
    let (slots, units, first) =
        model_runner.divide(&client, device, &host, model.as_ref(), &layout, &regions)?;

    // A rank whose share holds fewer slots than the deployment runs serves with those: the
    // storages its slots take are sized by them from here on. Its buckets and its pools are the
    // deployment's, which the slots do not change.
    let (model_runner, meta_config, meta_cache, meta_state) = if slots == model_runner.slots() {
        (model_runner, meta_config, meta_cache, meta_state)
    } else {
        let model_runner = model_runner.with_slots(slots);
        let (config, cache, state) = meta_owners(&model_runner, model.as_ref(), &layout)?;
        (model_runner, config, cache, state)
    };

    tracing::info!(
        rank = model_runner.rank(),
        group = model_runner.group(),
        slots,
        units,
        first,
        storage = %storage.socket,
        "the rank loaded its model and divided its device's quota"
    );
    Ok(LoadedRank {
        model_runner,
        client: Some(client),
        store: Some(store),
        weights,
        pool: None,
        streams: Vec::new(),
        keys,
        kernels,
        loaded: Loaded::Text(Box::new(TextRank {
            model,
            units,
            first,
            peaks,
            vision,
            tensor,
            config: meta_config,
            cache: meta_cache,
            state: meta_state,
            runner: None,
            vision_runner: None,
            union,
        })),
    })
}

/// The meta owners of a text rank's cache over `layout` and of its recurrent state, at the fewest
/// units the cache serves with, and the config of its runner of `model` over them, as
/// `model_runner` sizes them (steps 6-7): what the rank's profile run computes over, and what its
/// statement reads until its opening.
///
/// # Errors
///
/// What the runner's config or the owners refuse.
#[expect(
    clippy::type_complexity,
    reason = "the profiled runner's three owners, which `TextRank` holds apart"
)]
fn meta_owners(
    model_runner: &ModelRunner,
    model: &dyn TextModel,
    layout: &Arc<CacheLayout>,
) -> Result<(
    CausalLmConfig,
    Arc<Mutex<KvCacheMemory>>,
    Option<Arc<Mutex<StateMemory>>>,
)> {
    let least = fewest_units(layout);
    let config = model_runner.causal_config(model, layout, least)?;
    let cache = Arc::new(Mutex::new(KvCacheMemory::meta(model_runner.cache_config(
        Arc::clone(layout),
        least,
        None,
    ))?));
    let state = match model_runner.state_config(model, model_runner.buckets()) {
        Some(state) => Some(Arc::new(Mutex::new(StateMemory::meta(
            state,
            service::lock(&cache).config(),
        )?))),
        None => None,
    };
    Ok((config, cache, state))
}

/// The peaks of a text rank's runners, in the order of their ids: its causal runner's `peaks`,
/// then the `vision` runner's of a rank that encodes images.
fn runner_peaks(peaks: Peaks, vision: Option<Peaks>) -> Vec<Peaks> {
    std::iter::once(peaks).chain(vision).collect()
}

/// The device bytes of the region of encoder outputs of a rank that encodes `model`'s images of
/// at most `rows` rows (`VisionRunner::region_units`), which its media memory takes from the
/// resident pool.
///
/// # Errors
///
/// A model without a vision tower, no rows, or a region of more units than a session numbers.
fn encoder_bytes(model: &dyn TextModel, rows: u32) -> Result<u64> {
    let units = VisionRunner::region_units(model, rows)?;
    Ok(BufferMemory::device_bytes(units, 0, 0))
}

/// Loads a rank of a media deployment onto `backing` (steps 3-10), over `client`, its Storage
/// Server client (`None` on the meta device), on `device`:
///
/// - the text encoders' tensor groups and the denoisers' sequence groups over the process's host
///   group `host`, each a block's ranks (`ModelRunner::media_collectives`);
/// - the meta load of the stages its group places and its runners' profile run
///   (`ModelRunner::profile_media`), which states their peaks, buffers and kernels, a slot's
///   row and on a denoising rank the products' sizes;
/// - on a device, the generation its kernels resolve to and its weights, retained.
///
/// A media rank keeps no cache, so its division waits for its opening, where the session states
/// its product buffer: it only checks that the rank's regions fit.
fn load_video_rank(
    model_runner: ModelRunner,
    device: DeviceId,
    client: Option<Arc<StorageClient>>,
    host: &HostGroup,
    backing: Backing<'_>,
) -> Result<LoadedRank> {
    // Step 3: the parallel groups' communicators, opened before any device measurement because
    // NCCL keeps device buffers. Every rank forms both kinds, in this order.
    let (tensor, sequence) = model_runner.media_collectives()?;
    let communicates = client.is_some();
    let tensor = group_over(host, device, &tensor, communicates)
        .context("forming the text encoder's tensor group")?;
    let sequence = group_over(host, device, &sequence, communicates)
        .context("forming the denoiser's sequence group")?;

    // Steps 4-8: the meta load and the runners' profile run.
    let meta = model_runner
        .load_video(Weights::Meta)
        .context("loading the epoch's pipeline from its checkpoint's headers")?;
    let tasks = u32::try_from(media_tasks()).unwrap_or(u32::MAX);
    let (mut video, keys) = model_runner.profile_media(meta, (tensor, sequence), tasks)?;

    let (storage, resolve) = match backing {
        Backing::Serving { storage, kernels } => (storage, kernels),
        Backing::Meta => {
            return Ok(LoadedRank {
                model_runner,
                client: None,
                store: None,
                weights: Vec::new(),
                pool: None,
                streams: Vec::new(),
                keys,
                kernels: Arc::new(Generation::new(FIRST_GENERATION, [])),
                loaded: Loaded::Video(Box::new(video)),
            });
        }
    };
    let client = client.context("a served rank holds its Storage Server client")?;

    // Step 9: the generation the rank serves, every kernel it launches resolved.
    let kernels = resolve(device, &keys).context("resolving the rank's kernels")?;

    // Step 10: the weights, retained at the Storage Server and mapped on the device.
    let store = Arc::new(WeightStore::new(Arc::clone(&client), device)?);
    video.model = model_runner
        .load_video(Weights::Retained(store.as_ref()))
        .context("loading the epoch's pipeline")?;
    let weights = store
        .finish()
        .context("publishing the granules of the epoch's packed weights")?;

    tracing::info!(
        rank = model_runner.rank(),
        group = model_runner.group(),
        storage = %storage.socket,
        "the media rank loaded its pipeline"
    );
    Ok(LoadedRank {
        model_runner,
        client: Some(client),
        store: Some(store),
        weights,
        pool: None,
        streams: Vec::new(),
        keys,
        kernels,
        loaded: Loaded::Video(Box::new(video)),
    })
}

/// The hook through which union member `place` asks every other member to join its step: a
/// join batch to each one's worker, of the group that worker serves, once [`spawn`] set the
/// members' workers in `targets`.
fn join_hook(targets: Members, place: usize) -> JoinHook {
    Arc::new(move |batch: Batch| {
        let members = targets.get().ok_or(crate::worker::Error::State(
            "the expert union's workers are not spawned",
        ))?;
        for (index, worker) in members.iter().enumerate() {
            if index == place {
                continue;
            }
            let worker = worker.upgrade().ok_or(crate::worker::Error::State(
                "a worker of the expert union is closed",
            ))?;
            let mut join = batch.clone();
            join.group = worker.group();
            worker.join(join)?;
        }
        Ok(())
    })
}

/// The communicator of the collective group of `collective` this rank is a member of, absent for
/// a group of one and for a start that `communicates` none.
///
/// A group that communicates forms its NCCL communicator: each group's first member draws the
/// group's id and the host group carries it, one broadcast per group of more than one member,
/// which every member of the host group takes part in, in the order the groups are stated;
/// `Communicator::init` then blocks until every member of the rank's own group joined. One that
/// does not states the split alone, which the runner's graphs take
/// ([`Graphs::set_group`]).
fn group_over(
    host: &HostGroup,
    device: DeviceId,
    collective: &Collective,
    communicates: bool,
) -> Result<Option<Arc<ops::Group>>> {
    let members = host.size();
    for group in &collective.groups {
        if group.len() > 1 && group.iter().any(|member| *member >= members) {
            bail!(
                "a parallel group of members {group:?} in a host group of {members}: a group's \
                 ranks are members of the host group"
            );
        }
    }
    let mine = collective.member.filter(|(index, _)| {
        collective
            .groups
            .get(*index)
            .is_some_and(|group| group.len() > 1)
    });
    if !communicates {
        return Ok(None);
    }

    let mut own = None;
    for (index, group) in collective.groups.iter().enumerate() {
        let Some(&root) = group.first().filter(|_| group.len() > 1) else {
            continue;
        };
        let id = if host.rank() == root {
            Some(platform::UniqueId::new()?)
        } else {
            None
        };
        let bytes = host.broadcast_bytes(id.as_ref().map(|id| &id.0[..]), root)?;
        if mine.is_some_and(|(held, _)| held == index) {
            own = Some(platform::UniqueId(bytes.try_into().map_err(
                |bytes: Vec<u8>| {
                    anyhow::anyhow!("a communicator's id is 128 bytes, not {}", bytes.len())
                },
            )?));
        }
    }
    let (Some((index, rank)), Some(id)) = (mine, own) else {
        return Ok(None);
    };
    let size = collective.groups[index].len();
    let size = i32::try_from(size).context("a group wider than NCCL addresses")?;
    let member = i32::try_from(rank).context("a rank past what NCCL addresses")?;
    let communicator = platform::Communicator::init(device, size, member, &id)
        .context("joining a communicator")?;
    Ok(Some(Arc::new(ops::Group::new(communicator)?)))
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

/// The device memory the ranks of a media deployment's groups address: every rank's product
/// buffer and the host parts of its one process, which each rank reaches
/// (`BufferMemory::connect`), named by the devices of the deployment's serving ranks. Every group
/// of every layout over those devices states this one memory, however many groups a layout
/// places on them: the buffer pool numbers a deployment's memories with a bit each.
fn media_memory(groups: &[WorkerGroupConfig]) -> String {
    let mut devices: Vec<String> = groups
        .iter()
        .filter(|group| group.role.serves_requests())
        .flat_map(|group| group.ranks.iter())
        .map(|rank| format!("{}/{}", rank.host, rank.device))
        .collect();
    devices.sort_unstable();
    devices.dedup();
    devices.join(",")
}

/// The name of the device region of scheduler `scheduler`'s KV units of `layout`: two epochs of one
/// layout name one region, and any other layout another, whose memory space is its own
/// (`specs/buffer-pool.md` §14.1). The name covers what lays a unit out, never how many units
/// there are: an epoch numbers its cache from what its GPU's quota holds, a figure measured at
/// its start, and a successor of the same layout that measures another figure still names its
/// predecessor's region, whose numbering the buffer pool keeps (`Opened::limits`).
fn units_region(scheduler: usize, layout: &CacheLayout) -> Result<String> {
    let mut digest = storage::key::Digest::new(b"uniserve.units-region");
    for group in 0..layout.groups.len() {
        let shape = host::page_shape(layout, group)?;
        digest.bytes(&shape.layout.0);
    }
    digest.numbers(&[
        layout.unit_bytes,
        layout.plane_bytes,
        u64::from(layout.columns),
    ]);
    let digest = digest.finish();
    Ok(format!("kv {scheduler} {}", short(&digest)))
}

/// The leading eight bytes of `digest` in hex: what a region's name carries of the digest of
/// what lays its units out.
fn short(digest: &[u8; 32]) -> String {
    digest[..8]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// The layout the engine epoch `launch` states to open its session with its host's buffer pool
/// (`storage::runtime::Open`, `specs/buffer-pool.md` §11, §14.1), from `ranks`, this process's
/// ranks loaded to their division: [`text_layout`] for the ranks of a text deployment, from the
/// first rank's cache layout and units, its group's width, its request slots and their bytes,
/// and its model when it has a vision tower whose images the engine admits, which every rank of
/// the deployment's one configuration states alike; [`media_layout`] for the ranks of a media
/// deployment. It is the one statement of a session's layout, whoever builds the ranks behind
/// it.
///
/// # Errors
///
/// No rank, or as the layout of the ranks' kind refuses.
pub(crate) fn layout(launch: &EngineArgs, ranks: &[LoadedRank]) -> Result<Open> {
    let first = ranks.first().context("this process loaded no rank")?;
    match &first.loaded {
        Loaded::Text(text) => {
            let cache = Arc::clone(service::lock(&text.cache).layout());
            let tensor = u32::try_from(first.model_runner.world_size())
                .context("a worker group wider than a session states")?;
            let slots = first.model_runner.slots();
            let slot_bytes = CausalLmRunner::slot_bytes(&text.config, text.model.as_ref())?
                / u64::from(slots.max(1));
            let images = first.model_runner.image_rows(text.model.as_ref());
            let vision = (images > 0).then_some((text.model.as_ref(), images));
            text_layout(
                launch,
                &cache,
                text.units,
                tensor,
                (slots, slot_bytes),
                vision,
            )
        }
        Loaded::Video(_) => media_layout(launch, ranks),
    }
}

/// The layout of a text deployment's session ([`layout`]), for ranks of the deployment's one
/// configuration whose caches are laid out as `cache` with `units` units, the padding unit
/// included, whose worker groups are `tensor` ranks wide, whose groups each hold `slots`
/// request slots, each taking `slot_bytes` of one rank's device memory, and, when the engine
/// admits images, whose model with a vision tower and the most rows of an image it encodes are
/// `vision` (`ModelRunner::image_rows`; `None` otherwise):
///
/// - **Regions.** Per scheduler, the device region of its KV units, the same units on every rank
///   and group of it, named by its layout alone and numbered `units`. It states no segment and
///   asks for no unit: its memory is the segments its ranks map or create once the session is
///   open, which the head grants, and its scheduler then grows it into them by `Resize`. A
///   region an earlier epoch named keeps that epoch's numbering, which the answer states. Per
///   class of a cache group, when the deployment keeps a host
///   tier, the host region of its pages, one page a unit. Per worker group, the region of its
///   slot rows, numbered from 1, the epoch's own. A unit's bytes are summed over the tensor ranks
///   that hold a part of it.
/// - **Encoder outputs.** When a group encodes images (a group that prefills, of a model with a
///   vision tower, [`encodes`]): per such group, the region of its encoder outputs, the epoch's
///   own memory of its ranks, the same units on every rank of it (the tower is whole on each,
///   and each holds every output whole), sized by `VisionRunner::region_units`; and the host
///   region of encoder outputs ([`host_encoders`]), numbered for the outputs of every request
///   slot of each such group, and as many again for requests that resume.
/// - **Classes**, in the numbering the worker uses: cache group `g` is class `g`, then the class
///   of encoder outputs when a group encodes images, sized per request, with its host copies in
///   the host region of encoder outputs; then one class per product kind in the order of the
///   product classes (`crate::batch::product::KINDS`), then the class of stored trajectories. A
///   text deployment holds no product and no trajectory: those classes name no region.
/// - **Groups**, numbered for the session ([`group_numbers`]), each with its memory, its slot
///   region and per class its device regions; **per scheduler** its groups; and the
///   deployment's peers.
///
/// The epoch states no trial: the launch states none.
///
/// # Errors
///
/// Worker groups no scheduler block holds, a layout the regions' numbering cannot state, or a
/// vision tower that sizes no image.
pub fn text_layout(
    launch: &EngineArgs,
    cache: &CacheLayout,
    units: u32,
    tensor: u32,
    (slots, slot_bytes): (u32, u64),
    vision: Option<(&dyn TextModel, u32)>,
) -> Result<Open> {
    let config = &launch.engine_config;
    let groups = &config.parallel.worker_groups;
    let (numbers, schedulers) = group_numbers(groups, config.parallel.data_parallel_size.get())?;
    let tensor = u64::from(tensor.max(1));
    let kv = &config.kv_cache;

    let mut regions: Vec<Region> = Vec::new();
    let region_index = |regions: &Vec<Region>| -> Result<u8> {
        u8::try_from(regions.len()).context("more regions than a session numbers")
    };

    // The listed groups that encode images, by their index in the deployment.
    let mut encoding: Vec<usize> = Vec::new();
    for id in schedulers.iter().flatten() {
        let index = numbers
            .iter()
            .position(|number| number == id)
            .context("a listed group is numbered")?;
        if vision.is_some() && encodes(groups[index].role) {
            encoding.push(index);
        }
    }
    // The largest encoder output the session holds, the units of each group's region of them
    // on one rank, and the tensors they hold.
    let images = match vision.filter(|_| !encoding.is_empty()) {
        Some((model, rows)) => Some((
            VisionRunner::output_bytes(model, rows)?,
            VisionRunner::region_units(model, rows)?,
            feature_tensors(model),
        )),
        None => None,
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
            let pages = pages_of(shape.page_bytes, group.units_per_page);
            let name = format!("host kv {index} {}", short(&shape.layout.0));
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

    // The host region of encoder outputs: the outputs of every request slot of each group that
    // encodes images, and as many again for requests that resume.
    let encoders_region = match &images {
        Some((largest, _, tensors)) => {
            let units = encoders::pool_pages(&[*largest], slots as usize)
                .saturating_mul(encoding.len() as u64);
            let index = region_index(&regions)?;
            regions.push(host_encoders(tensors, units)?);
            Some(index)
        }
        None => None,
    };

    // Per scheduler, the region of its KV units; per group, its slot rows and, on a group that
    // encodes images, its region of encoder outputs.
    let slot_bytes = slot_bytes.max(1);
    let mut stated: Vec<Group> = Vec::new();
    for (scheduler, listed) in schedulers.iter().enumerate() {
        let name = units_region(scheduler, cache)?;
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
            if let Some((_, units, _)) = images {
                let outputs = if encoding.contains(&index) {
                    let name = format!("encoders {} {}", launch.epoch, id.0);
                    let region = region_index(&regions)?;
                    regions.push(Region {
                        space: name.clone(),
                        name,
                        tier: Tier::Device,
                        unit_bytes: UNIT_BYTES * tensor,
                        part_bytes: UNIT_BYTES,
                        reserved: 0,
                        units,
                        limit: units,
                        segments: Vec::new(),
                    });
                    vec![region]
                } else {
                    Vec::new()
                };
                per_class.push(outputs);
            }
            per_class.extend((0..=product::KINDS.len()).map(|_| Vec::new()));
            stated.push(Group {
                id: *id,
                memory: memory(group),
                slots: slot_region,
                regions: per_class,
            });
        }
    }

    // Class `g` is cache group `g`; then the encoder outputs of a deployment that encodes
    // images; then the product kinds and the stored trajectories, which a text deployment sizes
    // per request and holds none of.
    let ranks_per_unit = u32::try_from(tensor).unwrap_or(u32::MAX);
    let kinds = product::KINDS.len();
    let mut classes: Vec<Class> = Vec::with_capacity(cache.groups.len() + kinds + 2);
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
    // An encoder output's bytes are its rows in order under every parallel configuration, so
    // its class states one encoding everywhere (`crate::memory::encoders`).
    if let Some((_, _, tensors)) = images {
        classes.push(Class {
            units: 0,
            host: encoders_region.map(|region| (region, 0)),
            state_in_slot: false,
            tensors,
            encoding: encoders::layout(),
        });
    }
    // A text deployment keeps no product on the host, so no host copy of these classes is
    // written or compared: they state no tensors and no encoding.
    classes.extend((0..=kinds).map(|_| Class {
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

/// The layout of a media deployment's session ([`layout`]), from `ranks`, every rank of its
/// media groups (`ModelRunner::media_ranks` refuses a group with ranks in another process):
///
/// - **Regions.** The host region of encoder outputs when a group encodes prompts, and the host
///   region of stored trajectories when a group denoises: the deployment's on the host, numbered
///   for the outputs or trajectories every such group can have stored at once. A block of either
///   lies whole in one rank's memory and has one host copy, so a unit is one part
///   (`crate::memory::encoders::PAGE_BYTES` and a latent page), the part of the process's first
///   rank, which every rank copies with (`BufferMemory::part_holder`). Each is named by its
///   class's tensors, the layout of its units and its numbering, not by the ranks of a layout, so
///   another epoch of any parallel configuration that numbers it alike finds what this one
///   stored; the buffer pool keeps a region's numbering under its name. Per group, the product
///   buffer of each of its ranks, the epoch's own, of [`UNIT_BYTES`] units, holding the blocks of
///   every request slot of each product kind the rank backs at the largest size the denoising
///   group states; then the region of its slot rows, a unit the trajectory row of each of its
///   denoising ranks.
/// - **Classes.** One per product kind in `crate::batch::product::KINDS` order, sized per request:
///   the conditioning with its host copies in the host encoder region and the digests its encoding
///   states, the others with none; then the class of stored trajectories, with its host copies in
///   the host latent region and the digests the denoiser states (`trajectory_digests`).
/// - **Groups.** Each with the process's media memory ([`media_memory`]) and its slot rows, and
///   per product class the regions its blocks lie in: for a group whose phase writes the
///   product, the product buffers of its ranks, or for the conditioning its first text encoder
///   rank's alone, an encoder output being one rank's; for a group whose phase reads it, the
///   regions of the groups of its scheduler that write it, so the scheduler maps the writer's
///   completions into the reader's tables. **Per scheduler** its groups, and the deployment's
///   peers.
///
/// A rank backs the products its group writes. The sizes come from a rank of the group that
/// denoises, which states them for every product kind (`Capacity::media_bytes`).
///
/// # Errors
///
/// No denoising rank in the process, worker groups no scheduler block holds, or a layout the
/// regions' numbering cannot state.
fn media_layout(launch: &EngineArgs, ranks: &[LoadedRank]) -> Result<Open> {
    let config = &launch.engine_config;
    let groups = &config.parallel.worker_groups;
    let (numbers, schedulers) = group_numbers(groups, config.parallel.data_parallel_size.get())?;
    let firsts = WorkerGroupConfig::first_ranks(groups);

    // The sizes the group that denoises states, and the digests of the classes with host copies.
    let (sizer, sizing) = sizing(ranks)?;
    let (tensors, encoding, pages) = sizing
        .trajectory
        .context("a denoising rank states its trajectory class")?;
    let conditioning = product_kind(product::CONDITIONING)?;
    let request_units = request_units(sizing)?;

    // The serving groups by their index in the deployment, each with the phases its ranks serve.
    let mut listed: Vec<(usize, GroupId, usize)> = Vec::new();
    for (scheduler, ids) in schedulers.iter().enumerate() {
        for id in ids {
            let index = numbers
                .iter()
                .position(|number| number == id)
                .context("a listed group is numbered")?;
            listed.push((scheduler, *id, index));
        }
    }
    let phases: Vec<Vec<Phase>> = groups.iter().map(group_phases).collect();
    let serves = |index: usize, phase: Phase| phases[index].contains(&phase);
    // The request slots of each group that serves `phase`, as its ranks hold them
    // ([`fit_media_slots`]).
    let serving = |phase: Phase| -> Result<Vec<u32>> {
        listed
            .iter()
            .filter(|(_, _, index)| serves(*index, phase))
            .map(|(_, _, index)| group_slots(ranks, &groups[*index]))
            .collect()
    };

    let mut regions: Vec<Region> = Vec::new();
    let region_index = |regions: &Vec<Region>| -> Result<u8> {
        u8::try_from(regions.len()).context("more regions than a session numbers")
    };
    // A host region's one part per unit: a block of it lies whole in one rank's memory, and its
    // one host copy in the part the process's first rank maps.
    let host_region = |name: String, part_bytes: u64, units: u64| -> Result<Region> {
        let units = u32::try_from(units.max(1)).context("a host region of more units than u32")?;
        Ok(Region {
            space: name.clone(),
            name,
            tier: Tier::Host,
            unit_bytes: part_bytes,
            part_bytes,
            reserved: 0,
            units,
            limit: units,
            segments: Vec::new(),
        })
    };
    let conditioning_digests = sizer.conditioning_digests(sizing.model.as_ref());

    // The host region of encoder outputs: the conditioning of every request slot of each group
    // that encodes prompts, and as many again for requests that resume.
    let encoding_slots = serving(Phase::TextEncode)?;
    let encoders_region = if encoding_slots.is_empty() {
        None
    } else {
        let largest = largest_block(sizing, conditioning);
        let units = encoding_slots
            .iter()
            .map(|slots| encoders::pool_pages(&[largest], *slots as usize))
            .fold(0, u64::saturating_add);
        let index = region_index(&regions)?;
        regions.push(host_encoders(&conditioning_digests.0, units)?);
        Some(index)
    };

    // The host region of stored trajectories: every trajectory of each denoising group's slots
    // a preemption saves, and those a trial keeps beside them.
    let denoising_slots = serving(Phase::Denoise)?;
    let latents_region = if denoising_slots.is_empty() {
        None
    } else {
        let units = denoising_slots
            .iter()
            .map(|slots| latent::host::pool_pages(slot_pages(pages, *slots)))
            .fold(0, u64::saturating_add);
        let mut digest = storage::key::Digest::new(b"uniserve.host-latents");
        digest.bytes(&tensors);
        digest.bytes(&latent::host::share(&pages, WHOLE_SAMPLES));
        digest.number(units);
        let name = format!("host latents {}", short(&digest.finish()));
        let index = region_index(&regions)?;
        regions.push(host_region(name, pages.page_bytes()?, units)?);
        Some(index)
    };

    // Per group, the product buffer of each rank and the kinds each backs, then its slot rows.
    let mut backing: Vec<(usize, [Vec<u8>; 10], u8)> = Vec::with_capacity(listed.len());
    for &(_, id, index) in &listed {
        let group = &groups[index];
        let slots = group_slots(ranks, group)?;
        let mut kinds: [Vec<u8>; 10] = Default::default();
        for position in 0..group.ranks.len() {
            let backs = backed_kinds(group, position)?;
            let units: u64 = backs.iter().map(|kind| request_units[*kind]).sum();
            let units = u32::try_from((units * u64::from(slots)).max(1))
                .context("a product buffer of more units than u32")?;

            let name = products_region(launch.epoch, firsts[index] + position);
            let region = region_index(&regions)?;
            regions.push(Region {
                space: name.clone(),
                name,
                tier: Tier::Device,
                unit_bytes: UNIT_BYTES,
                part_bytes: UNIT_BYTES,
                reserved: 0,
                units,
                limit: units,
                segments: Vec::new(),
            });
            for kind in backs {
                kinds[kind].push(region);
            }
        }

        // A slot's row on each denoising rank holds a trajectory; an accounting figure, since the
        // rows are memory of the ranks' own processes.
        let rows = ranks
            .iter()
            .filter(|rank| rank.model_runner.group() == group.id.0)
            .filter_map(|rank| rank.video().ok().map(|video| video.row))
            .filter(|row| *row > 0);
        let (row, holders) = rows.fold((0, 0_u64), |(row, holders), held| {
            (held.max(row), holders + 1)
        });
        let row = row.next_multiple_of(UNIT_BYTES).max(1);
        let name = format!("slots {} {}", launch.epoch, id.0);
        let slot_region = region_index(&regions)?;
        regions.push(Region {
            space: name.clone(),
            name,
            tier: Tier::Device,
            unit_bytes: row * holders.max(1),
            part_bytes: row,
            reserved: 1,
            units: slots,
            limit: slots + 1,
            segments: Vec::new(),
        });
        backing.push((index, kinds, slot_region));
    }

    // The product classes and the class of stored trajectories.
    let mut classes: Vec<Class> = Vec::with_capacity(product::KINDS.len() + 1);
    for kind in 0..product::KINDS.len() {
        let stated = (kind == conditioning)
            .then_some(conditioning_digests)
            .zip(encoders_region);
        classes.push(Class {
            units: 0,
            host: stated.map(|(_, region)| (region, 0)),
            state_in_slot: false,
            tensors: stated.map_or([0; 32], |((tensors, _), _)| tensors),
            encoding: stated.map_or([0; 32], |((_, encoding), _)| encoding),
        });
    }
    classes.push(Class {
        units: 0,
        host: latents_region.map(|region| (region, 0)),
        state_in_slot: false,
        tensors,
        encoding,
    });
    let trajectory =
        u8::try_from(classes.len() - 1).context("more classes than a session numbers")?;

    // Each group's regions per class: those it writes, or those the groups of its scheduler that
    // write what it reads write; and the one memory every group addresses.
    let memory = media_memory(groups);
    let mut stated: Vec<Group> = Vec::with_capacity(listed.len());
    for (&(scheduler, id, index), (_, kinds, slot_region)) in listed.iter().zip(&backing) {
        let mut per_class: Vec<Vec<u8>> = Vec::with_capacity(classes.len());
        for (kind, (_, writer, reader)) in product::KINDS.iter().enumerate() {
            let regions = if serves(index, *writer) {
                kinds[kind].clone()
            } else if serves(index, *reader) {
                listed
                    .iter()
                    .zip(&backing)
                    .filter(|((held, _, other), _)| *held == scheduler && serves(*other, *writer))
                    .flat_map(|(_, (_, written, _))| written[kind].iter().copied())
                    .collect()
            } else {
                Vec::new()
            };
            per_class.push(regions);
        }
        per_class.push(Vec::new());
        stated.push(Group {
            id,
            memory: memory.clone(),
            slots: *slot_region,
            regions: per_class,
        });
    }

    Ok(Open {
        version: VERSION,
        epoch: launch.epoch,
        trial: false,
        regions,
        classes,
        trajectory,
        groups: stated,
        schedulers,
        peers: config.kv_cache.peers.clone(),
    })
}

/// The rank of `ranks` that sizes a media deployment's products and trajectories, the first that
/// denoises, with its model runner.
///
/// # Errors
///
/// No rank of `ranks` denoises.
fn sizing(ranks: &[LoadedRank]) -> Result<(&ModelRunner, &VideoRank)> {
    ranks
        .iter()
        .find_map(|rank| match &rank.loaded {
            Loaded::Video(video) if video.trajectory.is_some() => {
                Some((&rank.model_runner, &**video))
            }
            _ => None,
        })
        .context(
            "no rank of this process denoises, and a media deployment's products and \
             trajectories are sized by the group that does",
        )
}

/// The bytes of the largest block of product kind `kind` (an index into
/// `crate::batch::product::KINDS`) that the denoising rank `sizing` states.
fn largest_block(sizing: &VideoRank, kind: usize) -> u64 {
    sizing
        .capacity
        .media_bytes
        .iter()
        .filter_map(|(_, _, bytes)| bytes.get(kind).copied())
        .max()
        .unwrap_or(0)
}

/// The product buffer units one request's blocks of each kind take, in
/// `crate::batch::product::KINDS` order, at the sizes the denoising rank `sizing` states: a block
/// at the largest size, as many blocks as the request declares (one splice tail per unit of the
/// longest timeline).
///
/// # Errors
///
/// A session that numbers no splice tail kind.
fn request_units(sizing: &VideoRank) -> Result<Vec<u64>> {
    let tail = product_kind(product::VIDEO_TAIL)?;
    Ok((0..product::KINDS.len())
        .map(|kind| {
            let blocks = if kind == tail {
                u64::from(sizing.units)
            } else {
                1
            };
            largest_block(sizing, kind).div_ceil(UNIT_BYTES).max(1) * blocks
        })
        .collect())
}

/// The request slots media group `group` holds: its ranks' among `ranks`, which every rank of a
/// media group is ([`fit_media_slots`]).
///
/// # Errors
///
/// A group none of whose ranks is among `ranks`.
fn group_slots(ranks: &[LoadedRank], group: &WorkerGroupConfig) -> Result<u32> {
    ranks
        .iter()
        .find(|rank| rank.model_runner.group() == group.id.0)
        .map(|rank| rank.model_runner.slots())
        .with_context(|| format!("this process holds no rank of media group {}", group.id.0))
}

/// Decides the request slots of every media group of `ranks`, this process's ranks of a media
/// deployment loaded to their division (`specs/engine-worker-port.md` §1, `request_slots`): the
/// deployment's `max_running_requests`, or the fewest any rank of the group holds when fewer
/// fit. `left(rank)` is the bytes the rank's share of its GPU leaves once its weights, its graph
/// and scratch pools and its runners' buffers are taken out
/// (`ModelRunner::media_left`), `None` for a rank backed by nothing, which holds the stated
/// slots. A rank's slot holds what one request holds on it: its trajectory row
/// (`DenoiserRunner::row_bytes`) on a denoising rank, and on any rank the blocks one request
/// takes of each product kind it backs, at the largest sizes the denoising rank states
/// ([`request_units`]): its media memory of that many slots (`BufferMemory::device_bytes`).
///
/// A rank whose group holds fewer slots than the deployment runs states them of its group and
/// sizes its host latent pages and its media memory by them; its buckets and its pools are the
/// deployment's, which the slots do not change.
///
/// # Errors
///
/// A rank whose share holds no request slot, naming the bytes one needs, or no denoising rank
/// among `ranks`.
pub(crate) fn fit_media_slots(
    ranks: &mut [LoadedRank],
    left: impl Fn(&LoadedRank) -> Result<Option<u64>>,
) -> Result<()> {
    let Some(first) = ranks.first() else {
        return Ok(());
    };
    let config = first.model_runner.engine_config();
    let stated = u32::try_from(config.max_running_requests.get())
        .context("more request slots than the protocol addresses")?;
    let groups = &config.parallel.worker_groups;
    let request_units = request_units(sizing(ranks)?.1)?;

    // Per group, the fewest slots any of its ranks holds.
    let mut decided: BTreeMap<String, u32> = BTreeMap::new();
    for rank in ranks.iter() {
        let Some(left) = left(rank)? else {
            continue;
        };
        let video = rank.video()?;
        let runner = &rank.model_runner;
        let group = groups
            .iter()
            .find(|group| group.id.0 == runner.group())
            .with_context(|| format!("the deployment states no worker group {}", runner.group()))?;
        let request: u64 = backed_kinds(group, runner.position()?)?
            .iter()
            .map(|kind| request_units[*kind])
            .sum();
        let slots = division::request_slots(stated, left, |slots| -> Result<u64> {
            let units = u32::try_from((request * u64::from(slots)).max(1))
                .context("a product buffer of more units than u32")?;
            Ok(BufferMemory::device_bytes(units, slots, video.row))
        })
        .with_context(|| format!("sizing the request slots of rank {}", runner.rank()))?;
        tracing::info!(
            rank = runner.rank(),
            left,
            buffers = video.buffers,
            slot = BufferMemory::device_bytes(
                u32::try_from(request).unwrap_or(u32::MAX),
                1,
                video.row
            ),
            slots,
            "the media rank's share holds its request slots"
        );
        decided
            .entry(runner.group().to_owned())
            .and_modify(|held| *held = (*held).min(slots))
            .or_insert(slots);
    }

    for rank in ranks.iter_mut() {
        if let Some(&slots) = decided.get(rank.model_runner.group())
            && slots != rank.model_runner.slots()
        {
            reslot(rank, slots)?;
        }
    }
    Ok(())
}

/// Makes media rank `rank` hold `slots` request slots: its model runner's, and what it states per
/// slot (`VideoRank::hold`). Its buckets, its pools and its kernels are the deployment's, which
/// the slots do not change.
fn reslot(rank: &mut LoadedRank, slots: u32) -> Result<()> {
    let Loaded::Video(video) = &mut rank.loaded else {
        bail!("rank {} is a text rank", rank.model_runner.rank());
    };
    video.hold(slots);
    rank.model_runner = rank.model_runner.with_slots(slots);

    tracing::info!(
        rank = rank.model_runner.rank(),
        group = rank.model_runner.group(),
        slots,
        "the media rank holds fewer request slots than the deployment runs"
    );
    Ok(())
}

/// The host region of encoder outputs of `units` units whose class's tensors `tensors` digests
/// (`storage::runtime::Class::tensors`): the deployment's on the host, its unit one part of
/// `crate::memory::encoders::PAGE_BYTES`, since an output lies whole in one rank's memory and
/// has one host copy, in the one part its process copies with. It is named by the outputs'
/// tensors, the layout of its units and its numbering, not by the ranks of a layout, so another
/// epoch of any parallel configuration that numbers it alike finds what this one stored; the
/// buffer pool keeps a region's numbering under its name.
///
/// # Errors
///
/// A region of more units than a session numbers.
fn host_encoders(tensors: &[u8; 32], units: u64) -> Result<Region> {
    let mut digest = storage::key::Digest::new(b"uniserve.host-encoders");
    digest.bytes(tensors);
    digest.bytes(&encoders::layout());
    digest.number(units);
    let name = format!("host encoders {}", short(&digest.finish()));

    let units = u32::try_from(units.max(1)).context("a host region of more units than u32")?;
    Ok(Region {
        space: name.clone(),
        name,
        tier: Tier::Host,
        unit_bytes: encoders::PAGE_BYTES,
        part_bytes: encoders::PAGE_BYTES,
        reserved: 0,
        units,
        limit: units,
        segments: Vec::new(),
    })
}

/// The name of the product buffer of the deployment's rank `rank` in epoch `epoch`: memory of
/// the rank's own process, which each epoch has its own of (`specs/buffer-pool.md` §14.1).
fn products_region(epoch: u64, rank: usize) -> String {
    format!("products {epoch} {rank}")
}

/// The index in the session `layout` of the product buffer of `rank`, a media rank.
fn product_buffer(layout: &Open, rank: &LoadedRank) -> Result<u8> {
    let runner = &rank.model_runner;
    let name = products_region(runner.launch().epoch, runner.rank() as usize);
    let index = layout
        .regions
        .iter()
        .position(|region| region.name == name)
        .with_context(|| format!("the session states no region {name}"))?;
    u8::try_from(index).context("more regions than a session numbers")
}

/// Opens every rank of `ranks` over the session `layout` opened and `opened` answered, whose
/// answer states each region's segment table (`Opened::segments`), each rank on a thread of its
/// own (steps 16-22):
///
/// - the segments of the rank's KV region: the table's, which the rank maps where their epoch
///   created them, then those its epoch creates for the rest of the region's units
///   ([`plane_segments`]);
/// - the graph pool with the scratch and resident pools, the cache over those segments, the
///   recurrent state, the scratch pages of paired forwards, and the serving runner in its region
///   of the graph pool with its shadow region beside it, profiled again so its buckets can be
///   captured;
/// - on a rank that encodes images, its media memory over its group's region of encoder outputs
///   ([`encoder_memory`]), which its causal runner is lent with a feature buffer, and its vision
///   runner in its own region of the graph pool, profiled again.
///
/// On the meta device the owners hold no device and the segments no memory, so the cache, the
/// state and the runner are made over the region's units again, as the profile run's were.
///
/// Returns, per KV region by name, the segments its ranks created, as `(segment, first unit,
/// units)`: the head grants each once every rank that maps the region holds it. The ranks of a
/// media deployment are opened by [`open_video_ranks`] and create none.
///
/// # Errors
///
/// A rank whose group the session does not state, ranks of one region that created different
/// segments, or the first rank's failure.
pub(crate) fn open_ranks(
    ranks: &mut [LoadedRank],
    layout: &Open,
    opened: &Opened,
) -> Result<Vec<(String, Segments)>> {
    if ranks
        .iter()
        .any(|rank| matches!(rank.loaded, Loaded::Video(_)))
    {
        open_video_ranks(ranks, layout, opened)?;
        return Ok(Vec::new());
    }
    let tables = &opened.segments;

    // Each rank's KV region, its numbering and its table, read before the ranks go to their
    // threads.
    let mut regions = Vec::with_capacity(ranks.len());
    for rank in ranks.iter() {
        let (_, kv, _) = stated_regions(rank, layout)?;
        let table = tables
            .get(usize::from(kv))
            .with_context(|| format!("the answer states no segment table of region {kv}"))?;
        let region = &layout.regions[usize::from(kv)];
        regions.push((region.name.clone(), (region.limit, table)));
    }

    // The client of the process's first rank that encodes images, through which every rank that
    // does maps the one part of the host region of encoder outputs the process copies with.
    let holder = ranks
        .iter()
        .find(|rank| matches!(&rank.loaded, Loaded::Text(text) if text.vision.is_some()))
        .and_then(|rank| rank.client.clone());
    let session = (layout, opened);

    let opened: Vec<Segments> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(ranks.len());
        for (index, (rank, (_, region))) in ranks.iter_mut().zip(&regions).enumerate() {
            let holder = holder.as_ref();
            let handle = std::thread::Builder::new()
                .name(format!("rank-open-{index}"))
                .spawn_scoped(scope, move || open_rank(rank, *region, session, holder))
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

/// Opens one rank over its KV region of `units` units, the padding unit included, as the session
/// numbers it, and the region's segment table `table` (steps 16-22, [`open_ranks`]), in the
/// session whose layout is `session` and whose answer is `opened`. The rank's cache takes the
/// region's numbering, which is its own division's count unless an earlier epoch of the layout
/// numbered the region otherwise. A rank that encodes images maps the part of the host region of
/// encoder outputs through `holder` ([`encoder_memory`]). Returns the segments it created, as
/// `(segment, first unit, units)`.
fn open_rank(
    rank: &mut LoadedRank,
    (units, table): (u32, &Segments),
    (session, opened): (&Open, &Opened),
    holder: Option<&Arc<StorageClient>>,
) -> Result<Segments> {
    let device = rank.device()?;
    let group = stated_group(rank, session)?;
    let LoadedRank {
        model_runner,
        client,
        pool: rank_pool,
        streams,
        kernels,
        loaded,
        ..
    } = rank;
    let Loaded::Text(rank) = loaded else {
        bail!("a media rank opens no cache");
    };
    let layout = Arc::clone(service::lock(&rank.cache).layout());
    if units != rank.units {
        tracing::info!(
            rank = model_runner.rank(),
            divided = rank.units,
            numbered = units,
            "the rank takes the numbering an earlier epoch gave its region"
        );
    }
    rank.units = units;
    let state_config = model_runner.state_config(rank.model.as_ref(), model_runner.buckets());
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
    // The class of a rank's encoder outputs, the request slots its media memory's slot rows
    // number, and the most rows of an image it encodes.
    let class = encoder_class(&layout)?;
    let slots = model_runner.slots();
    let images = model_runner.image_rows(rank.model.as_ref());

    // Step 16: the segments of the region the table states, mapped, and the rest, created as
    // far as the quota held them at the division. A segment's every part is whole granules of
    // the device's allocations; the meta device allocates nothing, and each unit is a whole
    // granule of it.
    let granularity = match client {
        Some(_) => granularity(device)?,
        None => layout.plane_bytes,
    };
    let (segments, created) = plane_segments(
        model_runner.launch().epoch,
        (units, rank.first),
        &layout,
        granularity,
        table,
    )?;
    let cache_config = model_runner.cache_config(Arc::clone(&layout), units, Some(segments));

    let Some(client) = client.clone() else {
        // Steps 18, 19 and 21-22 over owners that hold no device.
        let cache = Arc::new(Mutex::new(KvCacheMemory::meta(cache_config)?));
        let state = match state_config {
            Some(state) => Some(Arc::new(Mutex::new(StateMemory::meta(
                state,
                service::lock(&cache).config(),
            )?))),
            None => None,
        };
        let media = match rank.vision {
            Some(_) => Some(Arc::new(encoder_memory(
                model_runner.rank(),
                (session, opened),
                group,
                (class, slots),
                None,
            )?)),
            None => None,
        };
        let mut runner = model_runner.runner(
            Arc::clone(&rank.model),
            config.clone(),
            Graphs::meta(Arc::clone(kernels), inputs),
            None,
            &cache,
            state.as_ref(),
            None,
            member,
            kernels,
        )?;
        if let Some(memory) = &media {
            runner.set_features(Arc::clone(memory), class)?;
        }
        profile(&runner)?;
        let vision = match media {
            Some(memory) => {
                let inputs = VisionRunner::input_bytes(rank.model.as_ref(), images)?;
                let graphs = Graphs::meta(Arc::clone(kernels), inputs);
                let model = Arc::clone(&rank.model);
                Some(vision_runner(model, graphs, None, memory, (class, images))?)
            }
            None => None,
        };

        rank.config = config;
        rank.cache = cache;
        rank.state = state;
        rank.runner = Some(runner);
        rank.vision_runner = vision;
        return Ok(created);
    };

    // Step 17: the graph pool, with a region for each runner and the scratch and resident pools
    // it reserves; the resident pool holds the recurrent state and, on a rank that encodes
    // images, its region of encoder outputs.
    let peaks = runner_peaks(rank.peaks, rank.vision);
    let (_, scratch) = model_runner.pools(device, &layout, &peaks)?;
    let outputs = match rank.vision {
        Some(_) => encoder_bytes(rank.model.as_ref(), images)?,
        None => 0,
    };
    let resident = state_config
        .as_ref()
        .map_or(0, |state| state.bytes(&layout))
        + outputs;
    let regions: Vec<(RunnerId, u64)> = (0..)
        .zip(&peaks)
        .map(|(id, peaks)| (RunnerId(id), peaks.graph))
        .collect();
    let (_, fraction) = model_runner.fractions()?;
    let mut pool = GraphPool::open(
        Arc::clone(&client),
        device,
        fraction,
        &regions,
        scratch,
        resident,
    )?;

    // Step 18: the cache, whose segments the rank maps or creates at the Storage Server.
    let stream =
        Stream::new(device).with_context(|| format!("creating device {}'s stream", device.0))?;
    let scratch_bytes = scratch_bytes(&config, &cache_config);
    let mut cache = KvCacheMemory::open(cache_config, device, &*client, &stream)
        .context("opening the rank's KV cache")?;

    // Step 19: the recurrent state, from the resident pool.
    let state = match state_config {
        Some(state) => Some(Arc::new(Mutex::new(StateMemory::open(
            state,
            cache.config(),
            pool.allocator(),
            stream.clone(),
            Arc::clone(kernels),
        )?))),
        None => None,
    };

    // Step 20: the scratch page set of paired forwards, the pages the rows of a batch share,
    // from the scratch pool.
    if config.scratch_pages > 0 {
        cache.open_scratch(config.scratch_pages, Some((pool.allocator(), &stream)))?;
    }
    let cache = Arc::new(Mutex::new(cache));

    // On a rank that encodes images, its media memory over its group's region of encoder
    // outputs, from the resident pool.
    let media = match rank.vision {
        Some(_) => {
            let holder = holder.unwrap_or(&client);
            let memory = encoder_memory(
                model_runner.rank(),
                (session, opened),
                group,
                (class, slots),
                Some((pool.allocator(), holder.as_ref())),
            )?;
            Some(Arc::new(memory))
        }
        None => None,
    };

    // Steps 21-22: the serving runner in its region of the graph pool, its shadow region beside
    // it (the profile run's shadow figure less the scratch pages, which the scratch pool holds),
    // profiled again so its buckets can be captured.
    let graphs = Graphs::new(
        stream.clone(),
        Arc::clone(kernels),
        pool.region(RunnerId(0))?,
        inputs,
        rank.peaks.inputs,
        2,
    )?;
    let mut runner = model_runner.runner(
        Arc::clone(&rank.model),
        config.clone(),
        graphs,
        Some(stream.clone()),
        &cache,
        state.as_ref(),
        rank.tensor.clone(),
        member,
        kernels,
    )?;
    if let Some(memory) = &media {
        runner.set_features(Arc::clone(memory), class)?;
    }
    let shadow = rank.peaks.shadow.saturating_sub(scratch_bytes);
    crate::runner::Runner::graphs(&runner).set_shadow(pool.shadow(shadow)?)?;
    profile(&runner)?;
    streams.push(("device", stream));

    // The vision runner in its region of the graph pool, on a stream of its own, profiled again
    // so its buckets can be captured; it computes no paired forward and keeps no shadow region.
    let vision = match (media, rank.vision) {
        (Some(memory), Some(peaks)) => {
            let encoding = Stream::new(device)
                .with_context(|| format!("creating device {}'s stream", device.0))?;
            let graphs = Graphs::new(
                encoding.clone(),
                Arc::clone(kernels),
                pool.region(RunnerId(1))?,
                VisionRunner::input_bytes(rank.model.as_ref(), images)?,
                peaks.inputs,
                2,
            )?;
            let model = Arc::clone(&rank.model);
            let issued = Some(encoding.clone());
            let runner = vision_runner(model, graphs, issued, memory, (class, images))?;
            streams.push(("device", encoding));
            Some(runner)
        }
        _ => None,
    };

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
    *rank_pool = Some(Arc::new(Mutex::new(pool)));
    rank.runner = Some(runner);
    rank.vision_runner = vision;
    Ok(created)
}

/// The media memory of rank index `rank` that encodes images, in its group `group` of the
/// session `session` opened and `opened` answered, whose encoder outputs are of class `class`:
/// over the group's region of encoder outputs, the same units on every rank of the group, and
/// the group's slot rows of `slots` slots, which hold no trajectory, by the answer's numbers;
/// with the one part of the host region of encoder outputs its process copies with. On a device
/// (`Some` allocator) it is taken from the allocator's resident pool and maps that part through
/// the client given beside it, the client of the process's first rank that encodes images, so
/// every rank of the process maps the one part that rank's index names, and an output one group
/// stored is where every group of the process loads it; otherwise it holds no device and states
/// the part.
///
/// # Errors
///
/// A group the session states no region of encoder outputs for, a region the answer does not
/// number, or the resident pool's or the server's refusal.
fn encoder_memory(
    rank: u32,
    (session, opened): (&Open, &Opened),
    group: &Group,
    (class, slots): (u8, u32),
    device: Option<(&mut platform::Allocator, &StorageClient)>,
) -> Result<BufferMemory> {
    let number = |index: u8| -> Result<u8> {
        opened
            .regions
            .get(usize::from(index))
            .map(|(number, _)| *number)
            .with_context(|| format!("the answer numbers no region {index}"))
    };
    let outputs = group
        .regions
        .get(usize::from(class))
        .and_then(|regions| regions.first())
        .copied()
        .with_context(|| {
            format!(
                "the session states no region of encoder outputs of group {}",
                group.id.0
            )
        })?;
    let products = (
        number(outputs)?,
        session.regions[usize::from(outputs)].units,
    );
    let slot_rows = (number(group.slots)?, slots);
    let host = session
        .classes
        .get(usize::from(class))
        .and_then(|class| class.host)
        .map(|(region, _)| region);

    let Some((allocator, holder)) = device else {
        let part = match host {
            Some(region) => {
                let units = session.regions[usize::from(region)].units;
                Some((number(region)?, encoders::PAGE_BYTES, units))
            }
            None => None,
        };
        return Ok(BufferMemory::meta(
            rank,
            products,
            slot_rows,
            0,
            part.as_slice(),
        )?);
    };

    let mut memory = BufferMemory::open(allocator, rank, products, slot_rows, 0)
        .context("opening the rank's memory of encoder outputs")?;
    if let Some(region) = host {
        let name = &session.regions[usize::from(region)].name;
        encoders::map_part(&mut memory, holder, (number(region)?, name))?;
    }
    Ok(memory)
}

/// Opens the ranks of a media deployment over the session `layout` opened and `opened`
/// answered (steps 11-22 for a rank that keeps no cache):
///
/// 1. each rank, on a thread of its own, checks that its regions fit its GPU's quota and opens
///    its graph pool with its runners' regions and its media memory in the resident pool; the
///    process's first rank maps the one part of every host region the session states, which
///    every rank copies with ([`open_media_memory`]). Two ranks on one device each open their
///    own;
/// 2. the process's media memories are connected, so each rank reaches every other's product
///    buffer and the parts, and they share their hand-offs;
/// 3. each rank, on a thread of its own, builds its serving runners over its memory in their
///    regions of the graph pool, with their shadow regions, profiled again so their buckets can
///    be captured, and its media host over the process's one media pool and media runner.
///
/// On the meta device the memories hold no device and state their parts, and the runners
/// compute over graphs that hold none.
///
/// # Errors
///
/// A rank whose group or regions the session does not state, a region the quota cannot hold,
/// the server's refusal of a part, or the first rank's failure.
fn open_video_ranks(ranks: &mut [LoadedRank], layout: &Open, opened: &Opened) -> Result<()> {
    let first = ranks.first().context("this process loaded no rank")?;
    let pool = Arc::new(first.model_runner.media_pool()?);
    let media = Arc::new(media::MediaRunner::new(media::MediaConfig {
        ffmpeg: first.model_runner.engine_config().media.ffmpeg.clone(),
    }));
    let numbers: Vec<u8> = opened.regions.iter().map(|(number, _)| *number).collect();
    // The rank that maps the host regions' parts: the process's first, which every epoch of
    // the deployment on this host holds.
    let holder = first.model_runner.rank();

    // The product buffers of the process's ranks, which each rank's device reaches.
    let buffers = ranks
        .iter()
        .map(|rank| {
            let index = product_buffer(layout, rank)?;
            numbers
                .get(usize::from(index))
                .copied()
                .with_context(|| format!("the answer numbers no region {index}"))
        })
        .collect::<Result<Vec<u8>>>()?;
    each_rank(ranks, "media-memory", |rank| {
        open_media_memory(rank, layout, (&numbers, &buffers), holder)
    })?;

    // Every rank's device reaches the other ranks' product buffers and parts.
    let mut memories = Vec::with_capacity(ranks.len());
    for rank in ranks.iter_mut() {
        let Loaded::Video(video) = &mut rank.loaded else {
            bail!("a text rank holds no media memory");
        };
        memories.push(
            video
                .memory
                .take()
                .context("a media rank's memory was not opened")?,
        );
    }
    BufferMemory::connect(&mut memories)?;

    let opened: Vec<_> = ranks.iter_mut().zip(memories).collect();
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(opened.len());
        for (index, (rank, memory)) in opened.into_iter().enumerate() {
            let (pool, media) = (&pool, &media);
            let handle = std::thread::Builder::new()
                .name(format!("rank-runners-{index}"))
                .spawn_scoped(scope, move || {
                    open_media_runners(rank, Arc::new(memory), (pool, media))
                })
                .context("starting a rank's runners")?;
            handles.push(handle);
        }
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("a rank's runners panicked"))?
            })
            .collect::<Result<Vec<()>>>()
    })?;
    Ok(())
}

/// Runs `open` on every rank of `ranks`, each on a thread of its own named `what` and its index,
/// and returns the first failure.
fn each_rank(
    ranks: &mut [LoadedRank],
    what: &str,
    open: impl Fn(&mut LoadedRank) -> Result<()> + Sync,
) -> Result<()> {
    std::thread::scope(|scope| {
        let open = &open;
        let mut handles = Vec::with_capacity(ranks.len());
        for (index, rank) in ranks.iter_mut().enumerate() {
            let handle = std::thread::Builder::new()
                .name(format!("rank-{what}-{index}"))
                .spawn_scoped(scope, move || open(rank))
                .with_context(|| format!("starting a rank's {what}"))?;
            handles.push(handle);
        }
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("a rank's {what} panicked"))?
            })
            .collect::<Result<Vec<()>>>()
            .map(drop)
    })
}

/// Opens a media rank's memory over the session `layout` states, its regions by the answer's
/// `numbers` ([`open_video_ranks`], step 1): its product buffer, its group's slot rows with a
/// slot's trajectory row, and on the rank `holder`, the process's first, the one part of every
/// host region the session states; and the regions its device block tables map (the host
/// regions it copies to among them, the encoder outputs' when its product buffer holds its
/// group's conditioning, the stored trajectories' when it denoises) and, of `buffers`, the
/// product buffers of the process's ranks, those of the other ranks, which its device reaches.
fn open_media_memory(
    rank: &mut LoadedRank,
    layout: &Open,
    (numbers, buffers): (&[u8], &[u8]),
    holder: u32,
) -> Result<()> {
    let device = rank.device()?;
    let group = stated_group(rank, layout)?;
    let products = product_buffer(layout, rank)?;
    let LoadedRank {
        model_runner,
        client,
        pool,
        loaded,
        ..
    } = rank;
    let Loaded::Video(video) = loaded else {
        bail!("a text rank opens no media memory");
    };
    let number = |index: u8| -> Result<u8> {
        numbers
            .get(usize::from(index))
            .copied()
            .with_context(|| format!("the answer numbers no region {index}"))
    };

    // The rank's product buffer and its group's slot rows, as the session states them.
    let units = layout.regions[usize::from(products)].units;
    let slots = model_runner.slots();

    // The host regions the session states, and those the rank copies to: the encoder outputs'
    // when its product buffer holds its group's conditioning, the stored trajectories' when it
    // denoises.
    let classes = product_classes(0);
    let conditioning = usize::from(classes[product_kind(product::CONDITIONING)?]);
    let holds_conditioning = group.regions.get(conditioning) == Some(&vec![products]);
    let encoders = layout.classes[conditioning].host.map(|(region, _)| region);
    let latents = layout.classes[usize::from(layout.trajectory)]
        .host
        .map(|(region, _)| region);
    let pages = latent_pages(0, 0)?;

    // The holder's parts, one of each host region, which every rank of the process copies with.
    let holds_parts = model_runner.rank() == holder;
    let mut parts: Vec<(u8, u64, u32)> = Vec::new();
    if let Some(region) = encoders.filter(|_| holds_parts) {
        let units = layout.regions[usize::from(region)].units;
        parts.push((region, encoders::PAGE_BYTES, units));
    }
    if let Some(region) = latents.filter(|_| holds_parts) {
        let units = layout.regions[usize::from(region)].units;
        parts.push((region, pages.page_bytes()?, units));
    }

    // The regions the rank's tables map, and the other ranks' product buffers its device reaches.
    let mut mapped = vec![number(group.slots)?, number(products)?];
    let copied = [
        encoders.filter(|_| holds_conditioning),
        latents.filter(|_| video.row > 0),
    ];
    for region in copied.into_iter().flatten() {
        mapped.push(number(region)?);
    }
    let own = number(products)?;
    let peers = buffers
        .iter()
        .copied()
        .filter(|buffer| *buffer != own)
        .collect();
    video.regions = (mapped, peers);

    let rank_index = model_runner.rank();
    let products = (number(products)?, units);
    let slot_rows = (number(group.slots)?, slots);
    let Some(client) = client else {
        let parts = parts
            .iter()
            .map(|(region, bytes, units)| Ok((number(*region)?, *bytes, *units)))
            .collect::<Result<Vec<_>>>()?;
        video.memory = Some(BufferMemory::meta(
            rank_index, products, slot_rows, video.row, &parts,
        )?);
        return Ok(());
    };

    // Steps 11-14 and 17: the regions fit what the quota leaves, and the graph pool opens with
    // each runner's region, their shadow regions and the media memory in its resident pool.
    let granularity = crate::process::ranks::rank::memory::granularity(device)?;
    let graphs: Vec<u64> = video.peaks.iter().map(|(_, peaks)| peaks.graph).collect();
    let shadows: Vec<u64> = video.peaks.iter().map(|(_, peaks)| peaks.shadow).collect();
    let pools = division::pool_bytes(granularity, &graphs, &shadows, 0);
    let resident = BufferMemory::device_bytes(units, slots, video.row);
    model_runner.media_division(client, device, pools, resident, video.buffers)?;
    let regions: Vec<(RunnerId, u64)> = video
        .peaks
        .iter()
        .map(|(id, peaks)| (*id, peaks.graph))
        .collect();
    let (_, fraction) = model_runner.fractions()?;
    let mut graph_pool = GraphPool::open(
        Arc::clone(client),
        device,
        fraction,
        &regions,
        pools.1,
        resident,
    )?;

    let mut memory = BufferMemory::open(
        graph_pool.allocator(),
        rank_index,
        products,
        slot_rows,
        video.row,
    )
    .context("opening the rank's media memory")?;
    if let Some(region) = encoders.filter(|_| holds_parts) {
        let name = &layout.regions[usize::from(region)].name;
        encoders::map_part(&mut memory, client, (number(region)?, name))?;
    }
    if let Some(region) = latents.filter(|_| holds_parts) {
        let name = &layout.regions[usize::from(region)].name;
        latent::host::map_part(&mut memory, client, (number(region)?, name), &pages)?;
    }

    tracing::info!(
        rank = rank_index,
        group = model_runner.group(),
        units,
        "the media rank opened its memory"
    );
    *pool = Some(Arc::new(Mutex::new(graph_pool)));
    video.memory = Some(memory);
    Ok(())
}

/// Builds a media rank's serving runners over `memory`, its media memory joined with the
/// process's others ([`open_video_ranks`], step 3): each device runner in its region of the
/// graph pool with its shadow region, profiled again, and its media host over the process's
/// media pool and media runner `host`.
fn open_media_runners(
    rank: &mut LoadedRank,
    memory: Arc<BufferMemory>,
    host: (&Arc<ThreadPool>, &Arc<media::MediaRunner>),
) -> Result<()> {
    let device = rank.device()?;
    let LoadedRank {
        model_runner,
        pool,
        streams,
        kernels,
        loaded,
        ..
    } = rank;
    let Loaded::Video(video) = loaded else {
        bail!("a text rank builds no media runner");
    };
    let classes = product_classes(0);
    let model = Arc::clone(&video.model);
    let stream =
        || Stream::new(device).with_context(|| format!("creating device {}'s stream", device.0));

    let runners = match pool {
        None => boxed(model_runner.device_runners(
            &model,
            &memory,
            |_, inputs| Ok((Graphs::meta(Arc::clone(kernels), inputs), None)),
            (None, None),
            &classes,
        )?),
        Some(pool) => {
            let mut pool = service::lock(pool);
            let peaks = video.peaks.clone();
            let peak = |id: RunnerId| {
                peaks
                    .iter()
                    .find(|(held, _)| *held == id)
                    .map(|(_, peaks)| *peaks)
                    .with_context(|| format!("runner {} was not profiled", id.0))
            };
            let typed = {
                let pool = &*pool;
                model_runner.device_runners(
                    &model,
                    &memory,
                    |id, inputs| {
                        let stream = stream()?;
                        let graphs = Graphs::new(
                            stream.clone(),
                            Arc::clone(kernels),
                            pool.region(id)?,
                            inputs,
                            peak(id)?.inputs,
                            2,
                        )?;
                        streams.push(("device", stream.clone()));
                        Ok((graphs, Some(stream)))
                    },
                    (video.tensor.clone(), video.sequence.clone()),
                    &classes,
                )?
            };
            let runners = boxed(typed);
            for (id, runner) in &runners {
                let shadow = peak(*id)?.shadow;
                if shadow > 0 {
                    runner.graphs().set_shadow(pool.shadow(shadow)?)?;
                }
            }
            runners
        }
    };

    // Profiled again on the graphs they serve with, so their buckets can be captured.
    let mut serving: Vec<Box<dyn Runner>> = Vec::with_capacity(runners.len() + 1);
    for (id, runner) in runners {
        runner
            .graphs()
            .profile_run(runner.as_ref())
            .with_context(|| format!("profiling the rank's runner {}", id.0))?;
        serving.push(runner);
    }
    let copies = pool.as_ref().map(|_| stream()).transpose()?;
    if let Some(copies) = &copies {
        streams.push(("copy", copies.clone()));
    }
    if let Some(media) =
        model_runner.media_host(model.as_ref(), (&memory, copies), host, kernels, &classes)?
    {
        serving.push(Box::new(media));
    }
    video.runners = serving;
    Ok(())
}

/// Where one rank's part of a region's units lies, as it states it (`Place`): the region, the
/// device memory, the part (the rank's index in its worker group) and the runs of its fields.
pub(crate) type Place = (String, String, u32, Vec<UnitRun>);

/// Binds each rank of `ranks` to the session `layout` opened and `opened` answered: a served rank
/// takes its part of each host region at its own host's Storage Server (`HostPart`), every rank's
/// cache is told the numbers of the regions its copies name (`KvCacheMemory::set_regions`), and
/// each served rank states where its part of its scheduler's KV units lies in its group's memory.
/// A rank of the session's own host states it at its server itself (`Place`, with `place`); a
/// peer's rank lies on another host than the session's buffer pool, whose head states it there
/// for the rank (`PlaceRemote`). A media rank keeps no cache, and its memory was opened over the
/// answer's numbers.
///
/// Returns where each served rank's part of its KV region's units lies, as `(region, device
/// memory, part, runs)`, the part being the rank's index in its worker group: what a peer
/// answers the session with (`Placed`).
///
/// # Errors
///
/// A rank whose group the session does not state, or the Storage Server's refusal.
pub(crate) fn bind(
    ranks: &mut [LoadedRank],
    layout: &Open,
    opened: &Opened,
    place: bool,
) -> Result<Vec<Place>> {
    let mut places = Vec::new();
    for rank in ranks.iter_mut() {
        let Loaded::Text(text) = &rank.loaded else {
            continue;
        };
        let (group, kv, host) = stated_regions(rank, layout)?;
        let number = |index: u8| -> Result<u8> {
            opened
                .regions
                .get(usize::from(index))
                .map(|(number, _)| *number)
                .with_context(|| format!("the answer numbers no region {index}"))
        };

        let mut cache = service::lock(&text.cache);
        let mut copies = None;
        if let Some(client) = &rank.client {
            let region = &layout.regions[usize::from(kv)].name;
            let runs = unit_runs(&cache.info().plane_runs);
            let part = rank.model_runner.tensor_rank();
            if place {
                client
                    .place(region, &group.memory, part, &runs)
                    .with_context(|| {
                        format!(
                            "placing the rank's units of region {region} in {}",
                            group.memory
                        )
                    })?;
            }
            places.push((region.clone(), group.memory.clone(), part, runs));

            // The stores and loads the buffer pool asks of this rank copy between its units and
            // its part of each group's host region, unit `u` of the region being page `u`.
            if host.iter().any(Option::is_some) {
                let names: Vec<Option<&str>> = host
                    .iter()
                    .map(|region| {
                        region.map(|index| layout.regions[usize::from(index)].name.as_str())
                    })
                    .collect();
                let device = rank.device()?;
                let stream = Stream::new(device)
                    .with_context(|| format!("creating device {}'s copy stream", device.0))?;
                cache
                    .map_host_parts(
                        client,
                        (&names, rank.model_runner.tensor_rank()),
                        stream.clone(),
                        Arc::clone(&rank.kernels),
                    )
                    .context("mapping the rank's parts of the host KV regions")?;
                copies = Some(stream);
            }
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
        drop(cache);

        if let Some(stream) = copies {
            rank.streams.push(("copy", stream));
        }
    }
    Ok(places)
}

/// The session's statement of `rank`'s group, the index of its KV region, and per cache group the
/// index of its host region.
fn stated_regions<'a>(
    rank: &LoadedRank,
    layout: &'a Open,
) -> Result<(&'a Group, u8, Vec<Option<u8>>)> {
    let group = stated_group(rank, layout)?;
    let kv = group
        .regions
        .first()
        .and_then(|regions| regions.first())
        .copied()
        .context("the session states no KV region of the rank's group")?;
    let groups = service::lock(&rank.text()?.cache).layout().groups.len();
    let host = layout
        .classes
        .iter()
        .take(groups)
        .map(|class| class.host.map(|(region, _)| region))
        .collect();
    Ok((group, kv, host))
}

/// The session's statement of `rank`'s group.
fn stated_group<'a>(rank: &LoadedRank, layout: &'a Open) -> Result<&'a Group> {
    let id = session_group(rank, layout)?;
    layout
        .groups
        .iter()
        .find(|group| group.id == id)
        .with_context(|| format!("the session states no group {}", id.0))
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

/// Spawns one worker per rank of `ranks` over its runners (step 23), each a rank of its worker
/// group with the group's device block tables over the regions of the session `layout` opened
/// and `opened` answered, and the copy budget, stating its ended sequence to the group's
/// `completed` word where `registers` puts it; sets the expert union's join targets (step 24);
/// and routes the node's workers and `remotes`, the peer links with the groups each holds ranks
/// of, by worker group (step 27).
///
/// `registers` gives, for a group the session's schedulers list, where its `completed` word lies
/// in their queue pairs (`StorageClient::completed_registers`): the session's head maps them, and
/// a peer, which holds no queue pair, states its ended sequences to the head over its link
/// instead and maps none.
///
/// A group's ranks report to its sequences in the order `Route` states: this process's
/// workers of the group, then its peer links. The unit sizes the tables count a copy by are
/// every region's of the session, so every rank of a group starts the same copies with a batch.
///
/// # Errors
///
/// A rank whose runner was taken, registers that cannot be mapped, or a worker that cannot
/// start.
pub(crate) fn spawn(
    ranks: &mut [LoadedRank],
    (layout, opened): (&Open, &Opened),
    registers: &dyn Fn(GroupId) -> Result<Vec<(QueuePair, usize)>>,
    remotes: Vec<(Vec<GroupId>, Arc<Remote>)>,
) -> Result<(Arc<Route>, Vec<Arc<Worker>>)> {
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
            registers(*id)?
        } else {
            Vec::new()
        };
        sequences.push(Sequences::new(*id, local + links, registers));
    }

    let mut workers = Vec::with_capacity(ranks.len());
    for (rank, (id, place)) in ranks.iter_mut().zip(&placed) {
        let device = rank.pool.as_ref().map(|_| rank.model_runner.devices()[0]);
        let group = layout.groups.iter().find(|group| group.id == *id);
        let (tables, runners) = match &mut rank.loaded {
            Loaded::Text(text) => {
                let mapped = text_regions(layout, opened, group);
                let runner = text.runner.take().context(
                    "the rank's runner was spawned already, or the rank was never opened",
                )?;
                // The vision runner of a rank that encodes images is `RunnerId(1)`.
                let mut runners: Vec<Box<dyn Runner>> = vec![Box::new(runner)];
                if let Some(vision) = text.vision_runner.take() {
                    runners.push(Box::new(vision));
                }
                (BlockTables::new(*id, &mapped, &unit_bytes), runners)
            }
            // A media rank's tables keep the runs of a block in the other ranks' product buffers
            // too, which its device reaches.
            Loaded::Video(video) => {
                let (mapped, peers) = &video.regions;
                let tables = BlockTables::new(*id, mapped, &unit_bytes).with_peers(peers);
                let runners = std::mem::take(&mut video.runners);
                if runners.is_empty() {
                    bail!("the rank's runners were spawned already, or the rank was never opened");
                }
                (tables, runners)
            }
        };
        let group_sequences = sequences
            .iter()
            .find(|group| group.group() == *id)
            .context("every placed group has sequences")?;
        let worker = Worker::spawn(
            device,
            runners,
            Arc::clone(&rank.kernels),
            tables,
            Arc::clone(group_sequences),
            *place,
            COPY_BUDGET,
        )?;
        workers.push(Arc::new(worker));
    }

    // The union's members ask each other to join through their workers.
    let union = ranks.first().and_then(|rank| match &rank.loaded {
        Loaded::Text(text) => text.union.as_ref(),
        Loaded::Video(_) => None,
    });
    if let Some((_, targets, _)) = union {
        targets
            .set(workers.iter().map(Arc::downgrade).collect())
            .map_err(|_| anyhow::anyhow!("the expert union's workers were set already"))?;
    }

    let route = Route::new(sequences, workers.clone(), remotes)?;
    Ok((Arc::new(route), workers))
}

/// The regions a text rank of the session's group `group` maps, by the answer's numbers: the
/// group's slot rows, its regions of every class, and every host region a class copies to.
fn text_regions(layout: &Open, opened: &Opened, group: Option<&Group>) -> Vec<u8> {
    let Some(group) = group else {
        return Vec::new();
    };
    std::iter::once(group.slots)
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
        .collect()
}

/// What each worker group of `ranks`, this process's, and of the epoch's peers serves
/// (`Capacity`), by the session's number of the group: the statements of its ranks, joined
/// ([`join`]), with `stated_by_peers`, what each peer's ranks state per group, and `peers`, what
/// every peer's rank serves, in rank order. The ranks of a text group state alike; a media
/// group's ranks each state the phases of the blocks that hold them.
///
/// A cache group's page seed is `storage::key::page_seed` of its page identity
/// (`KvCacheMemory::page_identities`) for the identity of the weights every rank of the deployment
/// serves, in rank order (`crate::process::binding::page_identity`): what the weight binding keys
/// the ranks' pages by, so a block computed by other weights or in another layout is another
/// block.
///
/// A group's encoder identity is, in the same way, the identity of the weights of its ranks that
/// compute encoder outputs (a text encoding's conditioning, a chat image's features), in rank
/// order (`crate::process::binding::encoder_identities`): the encoder's and its vision tower's,
/// with whatever else those ranks hold, which keys the outputs (`storage::key::encoder_output`)
/// so that an output other weights computed is another block. A group none of whose ranks
/// encodes states none.
///
/// A group that denoises states the latent seed of the identities of the weights of every rank
/// of the deployment (`storage::key::latent_seed`), which keys the trajectories it stores
/// (`storage::key::latent_content`): a trajectory follows from the denoiser's weights and from
/// the encoders' that compute its conditioning. A weight update's `Flush` renews all three from
/// the same weights (`crate::process::operations`).
///
/// # Errors
///
/// A rank whose group the session does not state, or a capacity its config cannot state.
pub(crate) fn capacities(
    ranks: &[LoadedRank],
    layout: &Open,
    (peers, stated_by_peers): (&[Served], Vec<(GroupId, Capacity)>),
) -> Result<Vec<(GroupId, Capacity)>> {
    let mut served: Vec<Served> = Vec::with_capacity(ranks.len() + peers.len());
    for rank in ranks {
        let group = session_group(rank, layout)?;
        served.push(binding::served(group, rank.encodes(), &rank.weights));
    }
    served.extend_from_slice(peers);

    let identity = binding::page_identity(&served);
    let identities: Vec<String> = served
        .iter()
        .map(|(_, _, names)| binding::identity_of_names(names))
        .collect();
    let latent = storage::key::latent_seed(identities.iter().map(String::as_str));

    let mut stated: Vec<(GroupId, Capacity)> = Vec::new();
    for rank in ranks {
        let id = session_group(rank, layout)?;
        let capacity = match &rank.loaded {
            Loaded::Text(text) => {
                let cache = service::lock(&text.cache);
                let seeds = cache
                    .page_identities(&identity)
                    .iter()
                    .map(|identity| storage::key::page_seed(identity))
                    .collect();
                let model = text.model.as_ref();
                let vision = text
                    .vision
                    .map(|_| (model, rank.model_runner.image_rows(model)));
                rank.model_runner
                    .capacity(&text.config, cache.layout(), seeds, vision)?
            }
            Loaded::Video(video) => video.capacity.clone(),
        };

        match stated.iter_mut().find(|(group, _)| *group == id) {
            Some((_, group)) => join(group, &capacity),
            None => stated.push((id, capacity)),
        }
    }

    // What the peers' ranks state, joined over every rank of each group. A peer keyed its ranks'
    // pages by its own weights alone: every text rank lays its cache out alike, so a group's page
    // seeds are those this process's ranks computed for the weights of every rank.
    let seeds = stated
        .iter()
        .map(|(_, capacity)| &capacity.seeds)
        .find(|seeds| !seeds.is_empty())
        .cloned();
    for (id, capacity) in stated_by_peers {
        match stated.iter_mut().find(|(group, _)| *group == id) {
            Some((_, group)) => join(group, &capacity),
            None => stated.push((id, capacity)),
        }
    }

    let encoders = binding::encoder_identities(&served);
    for (id, capacity) in &mut stated {
        if let Some(seeds) = &seeds
            && !capacity.seeds.is_empty()
        {
            capacity.seeds.clone_from(seeds);
        }
        if let Some((_, identity)) = encoders.iter().find(|(group, _)| group == id) {
            capacity.encoder_identity.clone_from(identity);
        }
        if capacity.serves(Phase::Denoise) {
            capacity.latent_seed = latent;
        }
    }
    Ok(stated)
}

/// Joins what another rank of a group states into `group`, the group's statement: the group
/// serves every phase a rank serves, its stage of a phase holds the most requests a rank's
/// does, and a batch may take every shape a rank captured; the denoising steps and the
/// products' sizes are those of the ranks that denoise, which state them alike.
fn join(group: &mut Capacity, rank: &Capacity) {
    group.phases |= rank.phases;
    for (held, stated) in group.rows.iter_mut().zip(rank.rows) {
        *held = (*held).max(stated);
    }
    for shape in &rank.buckets {
        if !group.buckets.contains(shape) {
            group.buckets.push(*shape);
        }
    }
    if group.media_bytes.is_empty() {
        group.media_bytes.clone_from(&rank.media_bytes);
    }
    group.denoise_steps = group.denoise_steps.max(rank.denoise_steps);
    group.call_window = group.call_window.max(rank.call_window);
}

#[cfg(test)]
pub(crate) mod tests;

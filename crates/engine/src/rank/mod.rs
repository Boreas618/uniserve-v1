//! Each rank's model runner: how a main process loads, sizes and captures one rank of the epoch's
//! engine configuration.
//!
//! `crate::process` is the main process itself: the control thread, the device workers, the
//! schedulers and the hot update. Everything it needs *before* it can start a thread comes from an
//! epoch's configuration - which devices, which buckets are captured, how much of each device the
//! graphs may map, which blocking pools exist, which Program Server socket and targets this rank
//! speaks to - and none of that belongs in `proto::config::EngineConfig`: the config fixes the epoch
//! and must not learn how the worker assembles its devices, its buckets or its graphs. It also
//! does not hold every input, because half of what the main process needs is a property of the node
//! it runs on (`EngineArgs`) or of the devices it measured.
//!
//! So the derivation lives here, on the consumer's side of the daemon/engine seam: a
//! [`ModelRunner`] is one rank of the epoch's engine configuration resolved against the process's
//! arguments, and the accessors below are the whole of what the main process asks of it. It
//! dispatches by the kind of the family the checkpoint resolved to ([`Family`]) and by the
//! capabilities the loaded model implements; no concrete model is named here.
//!
//! The capacity plan - what the rank's resident memory is - is not here: it is resolved from a
//! measured device against the reported capacity, which is the capacity negotiation's own area
//! (`worker::memory::Memory::new`'s caller).

pub(crate) mod buckets;
mod memory;
mod placement;
mod report;
mod text;
mod tokens;
mod video;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use ops::DType;
use platform::DeviceId;
use program_server::Target;
use proto::request::GenerationLimits;
use worker::blocking::{BlockingKind, LaneLimits, Limits};
use worker::device::GraphStorage;
use worker::memory::Memory;
use worker::plan::Buckets;
use worker::stream::EventPool;

use buckets::{PrefillBound, expert_classes, tier_refusal};
use placement::Mapping;
use tokens::{checkpoint_tokens, understanding_limits};
use video::{SharedCodec, prompt_capacity};

use nn::family::{
    Exchange, ExpertOrder, Family, Split, TextModel, TextPlacement, VideoModel, VideoPlacement,
};
use proto::config::EngineArgs;

/// What a rank holds once its model is loaded: the model of its family's kind.
///
/// A caller that has a [`Loaded`] cannot ask it for a figure of the other kind, which a bare
/// `Arc<…>` would allow.
#[derive(Clone)]
pub(crate) enum Loaded {
    /// The causal language model, one `Arc` per rank: every device's runner reads the same weights.
    Text(Arc<dyn TextModel>),
    /// The video pipeline, whose blocks the runners take one `Arc` each.
    Video(Arc<dyn VideoModel>),
}

/// How the data plane loads this rank's model again: the rank's model runner, pointed at the
/// target.
pub(crate) struct RankModel {
    model_runner: ModelRunner,
}

impl RankModel {
    pub(crate) fn new(model_runner: ModelRunner) -> Self {
        Self { model_runner }
    }
}

impl worker::data_plane::ModelLoad for RankModel {
    fn load(
        &self,
        target: &worker::data_plane::Target,
        weights: &mut dyn nn::RetainedWeights,
    ) -> std::result::Result<(), String> {
        let mut model_runner = self.model_runner.clone();
        Arc::make_mut(&mut model_runner.launch)
            .engine_config
            .model
            .model = target.checkpoint.to_string_lossy().into_owned();
        model_runner
            .load_model(weights, &target.experts)
            .map(drop)
            .map_err(|error| format!("{error:#}"))
    }

    fn expert_layers(&self) -> BTreeMap<String, u32> {
        let Family::Text(family) = self.model_runner.family else {
            return BTreeMap::new();
        };
        family
            .open(self.model_runner.checkpoint())
            .map(|checkpoint| checkpoint.expert_layers())
            .unwrap_or_default()
    }
}

/// One rank's model runner: it loads the rank's model on its devices, sizes the KV cache from what
/// the device has left, captures the rank's CUDA graphs and builds its compute runners. The ranks
/// of a process share the process's launch arguments; the rest is each rank's own.
#[derive(Clone)]
pub(crate) struct ModelRunner {
    /// The family the checkpoint's declared class resolved to.
    family: Family,
    /// The output and the media lane the ranks of this process share, once the first rank made them.
    codec: Arc<std::sync::Mutex<Option<SharedCodec>>>,
    /// The process's arguments, which every rank of it shares: the epoch's engine configuration
    ///, the checkpoint, the epoch and the node's interfaces. A model runner that loads another
    /// checkpoint holds its own copy, whose configuration names that checkpoint.
    launch: Arc<EngineArgs>,
    /// This rank's devices, as the worker's names for them.
    devices: Vec<DeviceId>,
    /// The targets this rank resolves a module for.
    targets: Vec<Target>,
    /// The bucket catalog every device of this rank captures.
    buckets: Buckets,
    /// This rank's ordinal inside its group, which the storage claims and the cache plan's
    /// agreement with its group are keyed by.
    rank: u32,
    /// The epoch whose cache planes this rank maps, on a standby: it serves from the planes its
    /// predecessor serves from, whose segments that epoch created.
    cache_epoch: Option<u64>,
    /// The planes of the epoch `cache_epoch` names, as that epoch laid them out: what a standby
    /// maps is the serving rank's cache, which only that rank sized.
    adopted_cache: Option<ipc::CacheTable>,
    /// Ranks in the deployment, whose minimum the cache plan's block size and slot count are
    /// agreed by: the launch's, or this process's devices when its ranks are its devices.
    world_size: usize,
    /// This rank's part of the epoch's parallel shape, resolved with the model runner.
    mapping: Mapping,
    /// The expert union of this node's ranks, on a deployment whose experts are split.
    union: Option<Arc<worker::runners::experts::Union>>,
}

/// A rank's places in its deployment's parallel groups.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ParallelShape {
    /// Members of the denoiser's sequence group.
    pub(crate) sequence: usize,
    /// This rank's place in it.
    pub(crate) sequence_rank: usize,
    /// Members of the text encoder's tensor group.
    pub(crate) tensor: usize,
    /// This rank's place in it.
    pub(crate) tensor_rank: usize,
}

impl Default for ParallelShape {
    fn default() -> Self {
        Self {
            sequence: 1,
            sequence_rank: 0,
            tensor: 1,
            tensor_rank: 0,
        }
    }
}

/// The collective groups a rank's runners issue on, absent where one rank holds the block.
///
/// Two groups rather than one because the two blocks run on two streams of a device: a collective of
/// one must not queue behind the other's, and each communicator keeps one issue order.
#[derive(Clone, Default)]
pub(crate) struct RankGroups {
    /// The denoiser's sequence group.
    pub(crate) sequence: Option<Arc<nn::Group>>,
    /// The text encoder's tensor group.
    pub(crate) tensor: Option<Arc<nn::Group>>,
}

impl ModelRunner {
    /// The epoch `build` builds, with this node's arguments resolved against it.
    ///
    /// The devices and the Program Server socket are the node's, not the epoch's: the daemon
    /// starts every rank of a node with the ordinals that rank serves, and a main process that
    /// guessed its devices from the epoch's parallelism would capture graphs on another rank's
    /// device.
    pub(crate) fn of(launch: &EngineArgs, family: Family) -> Result<Self> {
        let build = &launch.engine_config;
        let mapping = Mapping::of(launch, build)?;
        if launch.devices.is_empty() {
            bail!(
                "rank {} of group {} was started with no device to serve",
                launch.rank,
                launch.group
            );
        }
        let mut devices = Vec::with_capacity(launch.devices.len());
        for ordinal in &launch.devices {
            let ordinal = i32::try_from(*ordinal).with_context(|| {
                format!("device ordinal {ordinal} is outside the driver's range")
            })?;
            devices.push(DeviceId(ordinal));
        }
        let rows: Option<Vec<u32>> = build
            .cuda_graph
            .decode_batch_sizes
            .as_ref()
            .map(|sizes| sizes.iter().map(|size| size.get()).collect());
        let tokens: Option<Vec<u32>> = build
            .cuda_graph
            .prefill_token_sizes
            .as_ref()
            .map(|sizes| sizes.iter().map(|size| size.get()).collect());
        // The bounds a batch may reach: requests in flight, and the context a prefill chunk
        // cannot exceed. A bucket above either would be captured and never filled.
        let rows_bound = u32::try_from(build.max_running_requests.get())
            .context("more request slots than the protocol addresses")?;
        // The leader resolved the context length from the checkpoint before it launched this
        // process; a configuration that states none was not resolved.
        if build.model.max_model_len == 0 {
            bail!("the engine configuration states no resolved `max_model_len`");
        }
        let tokens_bound = build.model.max_model_len;
        // A stated token tier above the bound that admits it is refused here, before the
        // catalogue is built and long before any weight is mapped; a stated decode tier above the
        // request bound is the clamp in `Scheduler::new`, not a refusal. `tier_refusal` is where
        // that asymmetry is stated and argued.
        if let Some(sizes) = &tokens {
            if let Some(refusal) = tier_refusal(rows.as_deref(), rows_bound, sizes, tokens_bound) {
                bail!("{refusal}");
            }
        }
        let buckets = Buckets::for_batch(
            rows.as_deref(),
            tokens.as_deref(),
            rows_bound,
            tokens_bound,
            true,
        )?;
        if buckets.is_empty() {
            bail!("the build config captures no graph bucket");
        }
        // The architecture-specific instruction set of every device the rank serves, each once.
        let mut targets: Vec<Target> = Vec::with_capacity(devices.len());
        for device in &devices {
            let target = Target::of(*device)
                .with_context(|| format!("reading device {}'s architecture", device.0))?;
            if !targets.contains(&target) {
                targets.push(target);
            }
        }
        Ok(Self {
            family,
            launch: Arc::new(launch.clone()),
            devices,
            targets,
            buckets,
            rank: launch.rank,
            cache_epoch: None,
            adopted_cache: None,
            world_size: launch.world_size as usize,
            codec: Arc::default(),
            mapping,
            union: None,
        })
    }

    /// The epoch's engine configuration.
    pub(crate) fn engine_config(&self) -> &proto::config::EngineConfig {
        &self.launch.engine_config
    }

    /// The checkpoint this rank serves.
    fn checkpoint(&self) -> &Path {
        Path::new(&self.launch.engine_config.model.model)
    }

    /// One model runner per rank this process serves (one main process drives every device of its
    /// node, one worker each).
    ///
    /// A media deployment's ranks are its devices: a rank holds its own partition of the model on
    /// its own device, so each is planned, loaded and opened as a rank of its own. The text class
    /// keeps one model runner, whose second device is refused where its runner is built.
    pub(crate) fn ranks(&self) -> Result<Vec<Self>> {
        let text = matches!(self.family, Family::Text(_));
        let attention = self.attention_ranks()?;
        // Dedicated expert ranks hold the experts between them, whatever the schedulers number.
        let expert_ranks = match attention {
            0 => self.mapping.expert_parallel,
            _ => self.devices.len() as u32 - attention,
        };
        let union_ranks = expert_ranks + attention;
        let tensor = self.mapping.tensor_parallel;
        if text && union_ranks == 1 && tensor == 1 {
            return Ok(vec![self.clone()]);
        }
        let count = self.devices.len();
        // A tensor group splits every layer of one model over this process's devices: each device
        // is a rank of its own, which loads its block of every weight on its own device and joins
        // the group's collectives, and the one scheduler of the group submits each batch to all of
        // them (`schedulers`).
        if text && tensor > 1 {
            if union_ranks > 1 {
                bail!(
                    "a deployment of {tensor} tensor ranks and {union_ranks} expert ranks in one \
                     process: a tensor group and an expert union of one process's devices are not \
                     served together"
                );
            }
            if count != tensor as usize {
                bail!(
                    "this node serves {count} devices and the deployment's tensor group has \
                     {tensor} ranks: a tensor group lives on one node's devices, one rank each"
                );
            }
            return Ok(self
                .devices
                .iter()
                .enumerate()
                .map(|(index, device)| {
                    let mut rank_runner = self.clone();
                    rank_runner.devices = vec![*device];
                    rank_runner.rank = self.rank + index as u32;
                    rank_runner.world_size = count;
                    rank_runner.mapping.tensor_rank = index as u32;
                    rank_runner
                })
                .collect());
        }
        // The schedulers of an expert-parallel deployment each hold one rank: its own cache and
        // attention, its share of the experts, and a member of the union that runs their steps. A
        // deployment of dedicated expert ranks puts its attention ranks first in the union.
        let union = text.then(|| {
            worker::runners::experts::Union::new(usize::try_from(union_ranks).unwrap_or(0))
        });
        if text && count != union_ranks as usize {
            bail!(
                "this node serves {count} devices and the deployment's experts are exchanged over \
                 {union_ranks} ranks: an expert union lives on one node's devices, one rank each"
            );
        }
        let capacity = self.exchange_capacity();
        Ok(self
            .devices
            .iter()
            .enumerate()
            .map(|(index, device)| {
                let mut rank_runner = self.clone();
                rank_runner.devices = vec![*device];
                rank_runner.rank = self.rank + index as u32;
                rank_runner.world_size = count;
                if text {
                    // An attention rank holds the first rank's window and computes nothing with
                    // it; `ExchangeViews::source_only` keeps it out of its forward.
                    rank_runner.mapping.union_rank = index as u32;
                    rank_runner.mapping.attention_ranks = attention;
                    rank_runner.mapping.expert_parallel = expert_ranks;
                    rank_runner.mapping.expert_rank = (index as u32).saturating_sub(attention);
                    rank_runner.union.clone_from(&union);
                    rank_runner.buckets = expert_classes(&rank_runner.buckets, capacity);
                }
                rank_runner
            })
            .collect())
    }

    /// The rows one rank's exchange carries at most: the largest batch its buckets hold.
    fn exchange_capacity(&self) -> u64 {
        self.buckets
            .iter()
            .map(|bucket| u64::from(bucket.tokens))
            .max()
            .unwrap_or(1)
    }

    /// Leading ranks of a deployment whose experts live on dedicated ranks, from its worker groups:
    /// the ranks of every group that is not an expert group, and zero without one.
    fn attention_ranks(&self) -> Result<u32> {
        let groups = &self.engine_config().parallel.worker_groups;
        let experts = |group: &&scheduler::WorkerGroupConfig| {
            group.role == scheduler::WorkerGroupRole::Experts
        };
        if !groups.iter().any(|group| experts(&group)) {
            return Ok(0);
        }
        let attention = groups.iter().filter(|group| !experts(group));
        u32::try_from(attention.map(|group| group.ranks.len()).sum::<usize>())
            .context("more attention ranks than the exchange addresses")
    }

    /// Whether this rank holds a scheduler: every rank but a dedicated expert rank does.
    pub(crate) fn schedules(&self) -> bool {
        self.mapping.attention_ranks == 0 || self.mapping.union_rank < self.mapping.attention_ranks
    }

    /// The expert union this rank is a member of, with its transport's shape.
    pub(crate) fn union_member(
        &self,
        device: DeviceId,
        model: &dyn TextModel,
    ) -> Result<Option<worker::runners::experts::Member>> {
        let Some(union) = &self.union else {
            return Ok(None);
        };
        let (shape, encoded) = self.transport_shape(device, model)?;
        let member = worker::runners::experts::Member::new(Arc::clone(union), shape);
        // An encoded block runs in the exchange's one fused launch.
        Ok(Some(if encoded { member.fused() } else { member }))
    }

    /// The shape of the transport a member of an expert union creates on `device`.
    ///
    /// Returns the shape and whether the experts are read packed.
    fn transport_shape(
        &self,
        device: DeviceId,
        model: &dyn TextModel,
    ) -> Result<(ops::moe::exchange::TransportShape, bool)> {
        let experts = model
            .experts()
            .context("an expert union exchanges routed experts, and this model has none")?;
        let shape = ops::moe::exchange::TransportShape {
            ranks: self.mapping.expert_parallel + self.mapping.attention_ranks,
            rank: self.mapping.union_rank,
            experts: experts.experts,
            hidden: model.hidden(),
            intermediate: experts.intermediate,
            max_tokens: u32::try_from(self.exchange_capacity())
                .context("more rows per rank than the exchange addresses")?,
            top: experts.top,
            dtype: model.dtype(),
            device: device.0,
            attention_ranks: self.mapping.attention_ranks,
        };
        Ok((shape, experts.encoded))
    }

    /// What this rank's expert transport takes of its device, for the capacity plan.
    fn transport_bytes(&self, device: DeviceId, model: &dyn TextModel) -> Result<u64> {
        if self.union.is_none() {
            return Ok(0);
        }
        let (shape, encoded) = self.transport_shape(device, model)?;
        Ok(shape.device_bytes(encoded))
    }

    /// The expert union this rank's worker answers wakes for.
    pub(crate) fn union(&self) -> Option<(&Arc<worker::runners::experts::Union>, usize)> {
        self.union
            .as_ref()
            .map(|union| (union, self.mapping.union_rank as usize))
    }

    /// This model runner on the cache planes of `epoch`, when this rank stands by for that epoch's process.
    pub(crate) fn sharing_cache_of(
        mut self,
        epoch: Option<u64>,
        table: Option<ipc::CacheTable>,
    ) -> Self {
        self.cache_epoch = epoch;
        self.adopted_cache = table;
        self
    }

    /// This rank as the one pool the scheduler plans against.
    ///
    /// The ranks of a group run one batch together, and `worker::batch::Workers` hands it to
    /// each of them, so a scheduler sees the deployment as one rank whose batches are the group's.
    pub(crate) fn scheduled(&self) -> Self {
        let mut rank_runner = self.clone();
        rank_runner.world_size = 1;
        // A scheduler of an expert union plans its one rank as a pool of its own.
        if rank_runner.union.is_some() {
            rank_runner.rank = 0;
        }
        rank_runner
    }

    /// The catalogue of prefill buckets this rank captures, and whether the deployment stated it.
    ///
    /// `Buckets::bounds` answers the largest prefill token count among the buckets this rank
    /// captures, skipping media computations whose extents are not a language-model batch size, so
    /// the figure is the ceiling the mechanism admits and the one a policy's limit is checked
    /// against - not the build config's, which may be wider than the bound that dropped it.
    pub(crate) fn prefill_catalogue(&self) -> PrefillBound {
        let (_, bound) = self.buckets.bounds();
        PrefillBound {
            bound,
            stated: self
                .engine_config()
                .cuda_graph
                .prefill_token_sizes
                .is_some(),
        }
    }

    /// The blocking pools' admission bounds, one lane per kind.
    ///
    /// The build config sizes no pool - a blocking lane's width follows the work it carries, not
    /// the epoch - so the bounds are derived from what the node has: the checkpoint's own reader
    /// count for the lane that maps weights and installs programs, and the host's parallelism for
    /// the lanes that read transfers, reclaim storage and handle media. A lane whose bound is
    /// narrower than the work in flight queues it; it never drops it.
    pub(crate) fn blocking_limits(&self) -> Limits {
        let available = std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(4);
        let shared = (available / 2).clamp(1, 8);
        let lanes = BlockingKind::ALL.map(|kind| match kind {
            BlockingKind::ModuleLoad => {
                let workers = self
                    .engine_config()
                    .load
                    .threads
                    .map_or_else(|| shared.clamp(1, 8), |threads| threads.get() as usize);
                LaneLimits {
                    workers,
                    capacity: workers.saturating_mul(2),
                }
            }
            _ => LaneLimits {
                workers: shared,
                capacity: shared.saturating_mul(4),
            },
        });
        Limits::new(lanes)
    }

    /// The Program Server's socket, which the rank's program client connects to.
    pub(crate) fn program_server(&self) -> &Path {
        Path::new(&self.launch.program_server)
    }

    /// This rank's ordinal inside its group.
    pub(crate) fn rank(&self) -> u32 {
        self.rank
    }

    /// This rank's devices, in the order the daemon named them.
    pub(crate) fn devices(&self) -> &[DeviceId] {
        &self.devices
    }

    /// The device's completion events: one pool per rank, shared by its workers.
    pub(crate) fn events(&self) -> Arc<EventPool> {
        EventPool::new()
    }

    /// How long a job may run without rank progress before it is reported `Stalled`: the
    /// launch file's when it states one apart from the build config, and the build config's else.
    pub(crate) fn watchdog_timeout(&self) -> Duration {
        self.launch
            .watchdog_timeout()
            .unwrap_or(self.engine_config().worker.watchdog_timeout)
    }

    /// Whether this rank keeps a host KV cache: a text rank of a deployment with a host tier.
    pub(crate) fn host_kv_cache(&self, loaded: &Loaded) -> bool {
        matches!(loaded, Loaded::Text(_))
            && self.engine_config().kv_cache.host_tier
            && self.engine_config().kv_cache.prefix_caching
    }

    /// What this rank's scheduler starts with ([`scheduler::SchedulerHandle::over_batches_telling`]):
    /// the process's arguments `launch`, with the engine configuration as this process resolves it,
    /// and the mechanism's limits for this rank ([`scheduler::SchedulerConfig::of`]).
    ///
    /// The configuration takes the facts of the model this rank serves, read where they are
    /// stated: the control tokens from the checkpoint (`checkpoint_tokens`), since a launch file
    /// states where the checkpoint is and not what its vocabulary means, and what one request may
    /// be given to generate from this class of model. It keeps the host KV tier only where
    /// `host_kv_cache` says this scheduler keeps one.
    ///
    /// The limits are what this rank captured: a causal-language rank admits up to the largest
    /// bucket of its catalog, whose sizes are the tiers a plan's batch must fit, and a media
    /// rank admits the deployment's requests, one computation each, at the widest prompt capacity it
    /// captures, with no tiers, since its buckets are not sized by tokens.
    pub(crate) fn scheduler_args(
        &self,
        launch: &EngineArgs,
        loaded: &Loaded,
        host_kv_cache: bool,
    ) -> Result<(EngineArgs, scheduler::SchedulerConfig)> {
        let running = self.engine_config().max_running_requests.get();
        let mut args = launch.clone();
        let model = &mut args.engine_config.model;
        let (max_batch, max_num_batched_tokens, graph_tiers) = match loaded {
            Loaded::Text(_) => {
                let (rows, tokens) = self.buckets.bounds();
                let Family::Text(family) = self.family else {
                    bail!("a text rank's family is a text family");
                };
                let end_of_text = family.open(self.checkpoint())?.end_of_text();
                let control = checkpoint_tokens(self.checkpoint(), end_of_text)?;
                // What one request may be given to generate on this model; see
                // `understanding_limits`.
                model.generation_limits = understanding_limits();
                model.bos = control.bos;
                model.eos = control.eos;
                // A causal LM encodes no image, so no token closes encoded image content, and `0`
                // stands for that absence rather than for an id.
                model.end_of_image = 0;
                let graph_tiers = self
                    .buckets
                    .iter()
                    .map(|bucket| bucket.tokens)
                    .collect::<std::collections::BTreeSet<u32>>()
                    .into_iter()
                    .collect();
                (rows as usize, tokens as usize, graph_tiers)
            }
            Loaded::Video(_) => {
                let tokens = prompt_capacity(&self.engine_config().video, model.max_model_len)?;
                // A media request generates a clip, not tokens, so no understanding feature is
                // served; the engine still needs the two lower bounds, as the frontend's arm for
                // this class states them (`crates/frontend/src/model/generation.rs`).
                model.generation_limits = GenerationLimits {
                    latent_downsample: 1,
                    max_cfg_branches: 1,
                    ..GenerationLimits::default()
                };
                // The media class decodes no tokens, so no tokenizer fact reaches a scheduler
                // call: `0` stands for the absence of a token, and the list of terminating tokens
                // is empty.
                model.bos = 0;
                model.eos = Vec::new();
                model.end_of_image = 0;
                (running, tokens as usize, Vec::new())
            }
        };
        model.canvas_sampling = None;
        model.prefill_outputs = true;
        if !host_kv_cache {
            args.engine_config.kv_cache.host_tier = false;
        }
        let limits = scheduler::SchedulerConfig::of(
            &args.engine_config,
            max_batch,
            max_num_batched_tokens,
            graph_tiers,
        );
        Ok((args, limits))
    }

    /// The Program Server's targets this rank resolves a module for.
    pub(crate) fn targets(&self) -> &[Target] {
        &self.targets
    }

    /// Loads the rank's model and opens the resident owners it leaves room for.
    ///
    /// The memory is opened after the load because the capacity plan measures a device the weights
    /// are already mapped on - what is left, not what was there before the load - so the caller
    /// holds no knowledge of the order the two must be done in.
    pub(crate) fn load_and_open(
        &self,
        weights: &mut dyn nn::RetainedWeights,
        storage: &mut dyn worker::memory::retained::ExclusiveStorage,
        host_group: &worker::parallel::host::HostGroup,
        experts: &ExpertOrder,
    ) -> Result<(Loaded, Memory, ipc::MainProcessInfo, GraphStorage)> {
        let loaded = self.load_model(weights, experts)?;
        let (memory, info, graph) = self.open_memory(&loaded, storage, host_group)?;
        Ok((loaded, memory, info, graph))
    }

    /// The model this rank's checkpoint holds, its weights reached through `weights` in the order
    /// the model's load asks for them. An update walks the same code, so what this function reads
    /// is exactly what a load of another checkpoint or another expert order would.
    pub(crate) fn load_model(
        &self,
        weights: &mut dyn nn::RetainedWeights,
        experts: &ExpertOrder,
    ) -> Result<Loaded> {
        let mapping = &self.mapping;
        match self.family {
            Family::Text(family) => {
                // The checkpoint's own representation stands unless the deployment fixed another,
                // which is the build config's `dtype` and `quantization`.
                let checkpoint = family.open(self.checkpoint())?;
                let opened = nn::Checkpoint::open(self.checkpoint())?;
                let mut loader = nn::WeightLoader::new(&opened, weights, None, None);
                let placement = TextPlacement {
                    tensor: Split {
                        size: mapping.tensor_parallel,
                        rank: mapping.tensor_rank,
                    },
                    experts: Split {
                        size: mapping.expert_parallel,
                        rank: mapping.expert_rank,
                    },
                    exchange: (self.union.is_some()).then_some(Exchange {
                        capacity: self.exchange_capacity(),
                        attention_ranks: mapping.attention_ranks,
                    }),
                };
                let model = checkpoint.load(&mut loader, &placement, experts)?;
                Ok(Loaded::Text(model))
            }
            Family::Video(_) => {
                // The deployment's own representation, which the daemon resolved against the
                // checkpoint before it wrote this build config (`dtype = "auto"` never reaches a
                // main process as `auto`).
                let checkpoint = self.open_video()?;
                let mut roots = vec![self.checkpoint().to_path_buf()];
                if let Some(base) = &self.engine_config().model.base {
                    roots.push(PathBuf::from(base));
                }
                let roots: Vec<&Path> = roots.iter().map(PathBuf::as_path).collect();
                let opened = nn::Checkpoint::open_roots(&roots)?;
                let mut loader = nn::WeightLoader::new(&opened, weights, None, None);

                // **The video pipeline's parallel shape is the deployment's, and the deployment
                // file is where it is stated.** The build config carries the tensor-parallel width
                // and no sequence degree; the page that placed this group states both beside the
                // block they split.
                let (sequence, tensor) = self.video_split(&*checkpoint)?;
                let placement = VideoPlacement {
                    components: self.capability_groups().map(|groups| {
                        groups
                            .iter()
                            .flat_map(|group| group.blocks.keys().cloned())
                            .collect()
                    }),
                    sequence,
                    tensor,
                    dtype: self.dtype(),
                    serving: self.video_serving()?,
                };
                let model = checkpoint.load(&opened, &mut loader, &placement)?;
                Ok(Loaded::Video(model))
            }
        }
    }

    /// The element type the deployment computes in, which the daemon resolved against the
    /// checkpoint before it wrote this build config.
    fn dtype(&self) -> DType {
        match self.engine_config().model.dtype {
            proto::request::ModelDtype::Float16 => DType::F16,
            proto::request::ModelDtype::BFloat16 => DType::BF16,
            proto::request::ModelDtype::Float32 => DType::F32,
        }
    }

    /// Batches this rank's schedulers keep in flight, which the capacity plan relays through (the
    /// launch's `queue_depth`).
    pub(crate) fn queue_depth(&self) -> usize {
        self.launch.queue_depth as usize
    }

    /// The ranks of this process's group; one process holds one pool, so it is the deployment's.
    pub(crate) fn world_size(&self) -> usize {
        self.world_size
    }
}

//! Each rank's model runner: how a main process loads, sizes and starts one rank of the epoch's
//! engine configuration.
//!
//! `crate::process` is the main process itself: the control thread, the workers, the
//! schedulers and the hot update. Everything it needs *before* it can start a thread comes from an
//! epoch's configuration - which devices, which buckets are captured, which thread pools exist,
//! which Build Server socket and targets this rank speaks to - and none of that belongs in
//! `proto::config::EngineConfig`: the config fixes the epoch and must not learn how the worker
//! assembles its devices, its buckets or its graphs. It also does not hold every input, because
//! half of what the main process needs is a property of the node it runs on (`EngineArgs`) or of
//! the devices it measured.
//!
//! So the derivation lives here, on the consumer's side of the daemon/engine seam: a
//! [`ModelRunner`] is one rank of the epoch's engine configuration resolved against the process's
//! arguments, and the accessors below are the whole of what the main process asks of it. It
//! dispatches by the model of the family the checkpoint resolved to ([`Family::model`]) and by the
//! capabilities the loaded model implements; no concrete model is named here.
//!
//! A rank starts in the order `specs/engine-worker-port.md` §1.1 states, which
//! `crate::process::ranks` drives: the model's meta load and the runner's profile run
//! ([`text`]), the cache's layout and the division of the GPU's quota
//! ([`memory`]), the device's owners and the serving runner
//! ([`text`]), and what the rank states of its group once its workers are ready
//! ([`report`]). A rank of a media deployment takes the same steps over the media
//! runners of the blocks its group places on it and its media memory, and keeps no cache
//! ([`video`]).

pub(crate) mod buckets;
pub(crate) mod memory;
mod placement;
pub(crate) mod report;
pub(crate) mod text;
pub(crate) mod video;

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use crate::parallel::host::HostGroup;
use crate::runner::buckets::Buckets;
use crate::runner::causal_lm::experts::{JoinHook, Member};
use crate::runner::thread_pool::ThreadPool;
use anyhow::{Context as _, Result, bail};
use build_server::Target;
use nn::Split;
use nn::family::{ExpertOrder, Family, Model, TextModel, TextPlacement};
use platform::DeviceId;
use proto::config::{EngineArgs, WorkerGroupConfig, WorkerGroupRole};

use buckets::tier_refusal;
use placement::Mapping;

/// Where a rank's model load puts its weights.
pub(crate) enum Weights<'a> {
    /// Retained storage: each weight is read from the checkpoint, converted and published.
    Retained(&'a dyn nn::RetainedWeights),
    /// Nowhere: each weight's element type and extents come from the shard headers, at a synthetic
    /// address ([`nn::WeightLoader::meta`]), for a profile run, which launches nothing.
    Meta,
}

impl<'a> Weights<'a> {
    /// The loader of `checkpoint` that puts its weights here, keeping the checkpoint's own
    /// representation: the launch states no precision of its own to the loader.
    fn loader<'c>(self, checkpoint: &'c nn::Checkpoint) -> nn::WeightLoader<'c>
    where
        'a: 'c,
    {
        match self {
            Self::Retained(store) => nn::WeightLoader::new(checkpoint, store, None, None),
            Self::Meta => nn::WeightLoader::meta(checkpoint, None, None),
        }
    }
}

/// How the weight binding loads this rank's model again: the rank's model runner, pointed at the
/// target.
pub(crate) struct RankModel {
    model_runner: ModelRunner,
}

impl RankModel {
    pub(crate) fn new(model_runner: ModelRunner) -> Self {
        Self { model_runner }
    }
}

impl crate::process::binding::ModelLoad for RankModel {
    fn load(
        &self,
        target: &crate::process::binding::Target,
        weights: &dyn nn::RetainedWeights,
    ) -> std::result::Result<(), String> {
        let mut model_runner = self.model_runner.clone();
        Arc::make_mut(&mut model_runner.launch)
            .engine_config
            .model
            .model = target.checkpoint.to_string_lossy().into_owned();
        let loaded = match model_runner.family.model {
            Model::Text(_) => model_runner
                .load_text(Weights::Retained(weights), &target.experts)
                .map(drop),
            Model::Video(_) => model_runner
                .load_video(Weights::Retained(weights))
                .map(drop),
        };
        loaded.map_err(|error| format!("{error:#}"))
    }

    fn expert_layers(&self) -> BTreeMap<String, u32> {
        let Model::Text(family) = self.model_runner.family.model else {
            return BTreeMap::new();
        };
        let quantization = &self.model_runner.engine_config().model.quantization;
        nn::Checkpoint::open(self.model_runner.checkpoint())
            .and_then(|checkpoint| family.open(&checkpoint, quantization))
            .map(|checkpoint| checkpoint.expert_layers())
            .unwrap_or_default()
    }
}

/// One rank's model runner: it loads the rank's model on its device, divides what the device's
/// quota leaves among the rank's storages and builds the rank's runner. The ranks of a process
/// share the process's launch arguments; the rest is each rank's own.
#[derive(Clone)]
pub(crate) struct ModelRunner {
    /// The family the checkpoint's declared class resolved to.
    family: Family,
    /// The process's arguments, which every rank of it shares: the epoch's engine configuration,
    /// the checkpoint, the epoch and the node's interfaces.
    launch: Arc<EngineArgs>,
    /// This rank's devices, as the worker's names for them.
    devices: Vec<DeviceId>,
    /// The request slots this rank holds: the deployment's `max_running_requests`, or the fewer
    /// the division of its GPU's share decides ([`ModelRunner::with_slots`]).
    slots: u32,
    /// The bucket catalog every device of this rank captures: the build config's sizes within the
    /// batch bound, with a speculating deployment's verification and drafter buckets. The bound is
    /// the deployment's running requests whatever slots the division decides: the scheduling
    /// policy's limits are stated against that catalogue, and a batch holds at most the group's
    /// slots whatever its buckets hold (`Capacity::holds`).
    buckets: Buckets,
    /// This rank's ordinal among the deployment's ranks (every worker group's, in the order the
    /// deployment states them), which the storage claims are keyed by.
    rank: u32,
    /// The worker group this rank serves in: a text rank's group of the deployment (its prefill or
    /// decode group, or one data-parallel replica).
    group: String,
    /// Ranks in this rank's worker group on every host.
    world_size: usize,
    /// This rank's part of the epoch's parallel shape, resolved with the model runner.
    mapping: Mapping,
    /// The members of the expert union of this node's ranks, on a deployment whose experts are
    /// split; `None` outside a union.
    union: Option<u32>,
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

/// The collective groups one kind of a deployment's parallel groups forms over the host group, and
/// this rank's place among them.
pub(crate) struct Collective {
    /// Each group's members, as members of the host group in the group's rank order: a block's
    /// ranks are any positions of its worker group, in the order the deployment states them.
    pub(crate) groups: Vec<Vec<usize>>,
    /// The group this rank is a member of and its rank in it, absent when it is in none.
    pub(crate) member: Option<(usize, usize)>,
}

impl ModelRunner {
    /// The epoch `launch` builds, with this node's arguments resolved against it.
    ///
    /// The devices and the Build Server socket are the node's, not the epoch's: the daemon
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
        // request bound only sizes a batch. `tier_refusal` is where that asymmetry is stated and
        // argued.
        if let Some(sizes) = &tokens
            && let Some(refusal) = tier_refusal(rows.as_deref(), rows_bound, sizes, tokens_bound)
        {
            bail!("{refusal}");
        }
        // A speculating deployment's decode steps are verifications, each prefill and
        // verification followed by its drafter's bucket.
        let buckets = Buckets::for_batch(
            rows.as_deref(),
            tokens.as_deref(),
            rows_bound,
            tokens_bound,
            true,
        )?
        .with_drafts(build.speculative.num_draft_tokens);
        if buckets.is_empty() {
            bail!("the build config captures no graph bucket");
        }
        Ok(Self {
            family,
            launch: Arc::new(launch.clone()),
            devices,
            slots: rows_bound,
            buckets,
            rank: launch.rank,
            group: launch.group.clone(),
            world_size: launch.world_size as usize,
            mapping,
            union: None,
        })
    }

    /// This rank holding `slots` request slots, as the division of its GPU's share decides
    /// (`crate::memory::division::request_slots`): the storages its slots take are sized by them,
    /// and its buckets are the deployment's.
    pub(crate) fn with_slots(&self, slots: u32) -> Self {
        Self {
            slots,
            ..self.clone()
        }
    }

    /// The epoch's engine configuration.
    pub(crate) fn engine_config(&self) -> &proto::config::EngineConfig {
        &self.launch.engine_config
    }

    /// The process's arguments.
    pub(crate) fn launch(&self) -> &EngineArgs {
        &self.launch
    }

    /// The checkpoint this rank serves.
    fn checkpoint(&self) -> &Path {
        Path::new(&self.launch.engine_config.model.model)
    }

    /// One model runner per rank this process serves (one main process drives every device of its
    /// node, one worker each), in rank order, each numbered and placed as the deployment places it
    /// on this process's host ([`placed_ranks`]).
    ///
    /// Each rank is a rank of the deployment's worker group that places it, or a member of the
    /// node's expert union when the deployment splits its experts.
    pub(crate) fn ranks(&self) -> Result<Vec<Self>> {
        let text = matches!(self.family.model, Model::Text(_));
        let attention = self.attention_ranks()?;
        // Dedicated expert ranks hold the experts between them, whatever the schedulers number.
        let expert_ranks = match attention {
            0 => self.mapping.expert_parallel,
            _ => self.devices.len() as u32 - attention,
        };
        let union_ranks = expert_ranks + attention;
        let tensor = self.mapping.tensor_parallel;
        if text && tensor > 1 && union_ranks > 1 {
            bail!(
                "a deployment of {tensor} tensor ranks and {union_ranks} expert ranks in one \
                 process: a tensor group and an expert union of one process's devices are not \
                 served together"
            );
        }
        // This process's ranks, each on the device that places it ([`placed_ranks`]).
        let placed = placed_ranks(self.launch())?;
        let count = placed.len();
        // Each rank is a rank of the deployment's worker group that places it: its prefill or
        // decode group, a data-parallel replica, or the one group of a tensor-parallel deployment.
        // A group's ranks split every layer of one model over the group's devices: each loads its
        // block of every weight on its own device and joins the group's collectives, and every
        // batch of the group is handed to all of them (`crate::executor::Route`), those of
        // another host's process through its peer link.
        if text && union_ranks == 1 {
            return placed
                .into_iter()
                .map(|(rank, _, _, device)| {
                    let (group, position, tensor) = self.placed(rank)?;
                    let mut rank_runner = self.clone();
                    rank_runner.devices = vec![device];
                    rank_runner.rank = rank;
                    rank_runner.group = group;
                    rank_runner.world_size = tensor as usize;
                    rank_runner.mapping.tensor_parallel = tensor;
                    rank_runner.mapping.tensor_rank = position;
                    Ok(rank_runner)
                })
                .collect();
        }
        // A media deployment's ranks are every rank of its worker groups, each on the device
        // that places it, which may hold ranks of several groups.
        if !text {
            return self.media_ranks();
        }
        // The schedulers of an expert-parallel deployment each hold one rank: its own cache and
        // attention, its share of the experts, and a member of the union that runs their steps. A
        // deployment of dedicated expert ranks puts its attention ranks first in the union.
        if count != union_ranks as usize {
            bail!(
                "this node serves {count} devices and the deployment's experts are exchanged over \
                 {union_ranks} ranks: an expert union lives on one node's devices, one rank each"
            );
        }
        let groups = &self.engine_config().parallel.worker_groups;
        placed
            .into_iter()
            .enumerate()
            .map(|(index, (rank, group, _, device))| {
                let mut rank_runner = self.clone();
                rank_runner.devices = vec![device];
                rank_runner.rank = rank;
                rank_runner.world_size = 1;
                // Each scheduler of the union drives its rank's own group.
                rank_runner.group.clone_from(&groups[group].id.0);
                // An attention rank holds the first rank's window and computes nothing with
                // it; `DispatchViews::source_only` keeps it out of its forward.
                rank_runner.mapping.union_rank = index as u32;
                rank_runner.mapping.attention_ranks = attention;
                rank_runner.mapping.expert_parallel = expert_ranks;
                rank_runner.mapping.expert_rank = (index as u32).saturating_sub(attention);
                rank_runner.union = Some(union_ranks);
                Ok(rank_runner)
            })
            .collect()
    }

    /// The worker group that places the deployment's rank `rank`, the rank's position in it and
    /// the group's tensor-parallel degree, which its ranks number.
    ///
    /// # Errors
    ///
    /// Fails when no group places the rank, or when the group's ranks are not one tensor group of
    /// its `model` block.
    fn placed(&self, rank: u32) -> Result<(String, u32, u32)> {
        let groups = &self.engine_config().parallel.worker_groups;
        let (index, position) =
            WorkerGroupConfig::place_rank(groups, rank as usize).with_context(|| {
                format!("the deployment's worker groups place no rank {rank} for this process")
            })?;
        let group = &groups[index];
        let tensor = group
            .blocks
            .get(proto::config::DEFAULT_BLOCK)
            .map_or(1, |block| block.tensor_parallel_size);
        if tensor != group.ranks.len() {
            bail!(
                "worker group {} places {} ranks and its `{}` block is {tensor}-way tensor \
                 parallel: a text group's ranks are one tensor group",
                group.id,
                group.ranks.len(),
                proto::config::DEFAULT_BLOCK
            );
        }
        Ok((
            group.id.0.clone(),
            u32::try_from(position).context("a group wider than the protocol addresses")?,
            u32::try_from(tensor).context("a group wider than the protocol addresses")?,
        ))
    }

    /// The worker group this rank serves in.
    pub(crate) fn group(&self) -> &str {
        &self.group
    }

    /// The tensor groups of a text deployment whose ranks are each one group's
    /// ([`Self::ranks`]), with this rank's place among them: every worker group of more than one
    /// rank, as consecutive members of the host group, whose members are the deployment's ranks.
    /// `None` for a member of an expert union, whose tensor group is the parallel shape's over
    /// the whole host group.
    ///
    /// # Errors
    ///
    /// As [`Self::placed`] fails for this rank.
    pub(crate) fn tensor_spans(&self) -> Result<Option<Collective>> {
        if !matches!(self.family.model, Model::Text(_)) || self.union.is_some() {
            return Ok(None);
        }
        let groups = &self.engine_config().parallel.worker_groups;
        let firsts = WorkerGroupConfig::first_ranks(groups);
        // A worker group's tensor group is its ranks in their order.
        let members: Vec<Vec<usize>> = firsts
            .iter()
            .zip(groups)
            .map(|(first, group)| (*first..*first + group.ranks.len()).collect())
            .collect();
        let (_, position, _) = self.placed(self.rank)?;
        let (group, _) =
            WorkerGroupConfig::place_rank(groups, self.rank as usize).with_context(|| {
                format!("the deployment's worker groups place no rank {}", self.rank)
            })?;
        Ok(Some(Collective {
            groups: members,
            member: Some((group, position as usize)),
        }))
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
        let experts = |group: &&WorkerGroupConfig| group.role == WorkerGroupRole::Experts;
        if !groups.iter().any(|group| experts(&group)) {
            return Ok(0);
        }
        let attention = groups.iter().filter(|group| !experts(group));
        u32::try_from(attention.map(|group| group.ranks.len()).sum::<usize>())
            .context("more attention ranks than the exchange addresses")
    }

    /// The member of the node's expert union this rank is on `device`, over the union's host
    /// group `group`, asking the other members to join its steps through `join`; `None` for a
    /// rank of no union.
    pub(crate) fn union_member(
        &self,
        device: DeviceId,
        model: &dyn TextModel,
        group: Arc<HostGroup>,
        join: JoinHook,
    ) -> Result<Option<Member>> {
        if self.union.is_none() {
            return Ok(None);
        }
        let (shape, encoded) = self.transport_shape(device, model)?;
        // An encoded block runs in the exchange's one fused launch.
        let mode = if encoded {
            ops::moe::exchange::Mode::Fused
        } else {
            ops::moe::exchange::Mode::Dispatch
        };
        Ok(Some(Member::new(group, shape, mode, join)?))
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

    /// What this rank's expert transport takes of its device, for the division.
    fn transport_bytes(&self, device: DeviceId, model: &dyn TextModel) -> Result<u64> {
        if self.union.is_none() {
            return Ok(0);
        }
        let (shape, encoded) = self.transport_shape(device, model)?;
        Ok(shape.device_bytes(encoded))
    }

    /// The expert union this rank is a member of: its members and this rank's place among them;
    /// `None` outside a union.
    pub(crate) fn union(&self) -> Option<(u32, u32)> {
        self.union.map(|members| (members, self.mapping.union_rank))
    }

    /// This rank's ordinal among the deployment's ranks.
    pub(crate) fn rank(&self) -> u32 {
        self.rank
    }

    /// This rank's devices, in the order the daemon named them.
    pub(crate) fn devices(&self) -> &[DeviceId] {
        &self.devices
    }

    /// The bucket catalog this rank captures.
    pub(crate) fn buckets(&self) -> &Buckets {
        &self.buckets
    }

    /// Batches the rank's groups keep in flight (the launch's `queue_depth`).
    pub(crate) fn queue_depth(&self) -> usize {
        self.launch.queue_depth as usize
    }

    /// The rank's index in its worker group, which names its part of the units of the session's
    /// regions at the Storage Server: the ranks of one index, in every group and replica, hold
    /// the same share of every unit.
    pub(crate) fn tensor_rank(&self) -> u32 {
        self.mapping.tensor_rank
    }

    /// The ranks of this rank's worker group on every host.
    pub(crate) fn world_size(&self) -> usize {
        self.world_size
    }

    /// The routed experts' distribution this epoch loads: what a redistribution committed, which
    /// a relaunch and a standby load again; empty for the checkpoint's own order.
    pub(crate) fn experts(&self) -> ExpertOrder {
        self.launch
            .expert_order
            .iter()
            .map(|(module, order)| (module.clone(), order.experts.clone()))
            .collect()
    }

    /// Whether this rank serves a media deployment: its family's model is a video pipeline.
    pub(crate) fn is_video(&self) -> bool {
        matches!(self.family.model, Model::Video(_))
    }

    /// The causal language model this rank's checkpoint holds, its weights reached through
    /// `weights` in the order the model's load asks for them. An update walks the same code, so
    /// what this function reads is exactly what a load of another checkpoint or another expert
    /// order would, and a meta load answers the same weights' element types and extents without
    /// reading them. A media rank's pipeline is [`ModelRunner::load_video`]'s.
    ///
    /// # Errors
    ///
    /// The checkpoint's or the family's refusal, and a checkpoint of a media family.
    pub(crate) fn load_text(
        &self,
        weights: Weights<'_>,
        experts: &ExpertOrder,
    ) -> Result<Arc<dyn TextModel>> {
        let mapping = &self.mapping;
        let Model::Text(family) = self.family.model else {
            bail!(
                "{} is a media model, which a text rank does not load",
                self.checkpoint().display()
            );
        };
        let opened = nn::Checkpoint::open(self.checkpoint())?;
        let checkpoint = family.open(&opened, &self.engine_config().model.quantization)?;
        let mut loader = weights.loader(&opened);
        let drafts = self.engine_config().speculative.num_draft_tokens;
        let placement = TextPlacement {
            tensor: Split {
                size: mapping.tensor_parallel,
                rank: mapping.tensor_rank,
            },
            experts: Split {
                size: mapping.expert_parallel,
                rank: mapping.expert_rank,
            },
            // The drafter's layers load when the deployment speculates with them, and a model
            // whose state keeps a window per request sizes it by the drafts.
            drafter: drafts > 0,
            draft_tokens: drafts,
        };
        Ok(checkpoint.load(&opened, &mut loader, &placement, experts)?)
    }
}

/// What the main process reads of a rank to run it: its thread pools and its devices' build
/// targets.
impl ModelRunner {
    /// The thread pool media host work runs on, one per process: half the host's parallelism
    /// within one to eight, each thread with [`MEDIA_TASKS`] tasks of room.
    pub(crate) fn media_pool(&self) -> Result<ThreadPool> {
        let threads = host_share();
        Ok(ThreadPool::new("media", media_tasks(), threads)?)
    }

    /// The Build Server's targets this rank resolves a module for: the architecture-specific
    /// instruction set of every device the rank serves, each once.
    ///
    /// # Errors
    ///
    /// When a device does not state its architecture.
    pub(crate) fn targets(&self) -> Result<Vec<Target>> {
        let mut targets: Vec<Target> = Vec::with_capacity(self.devices.len());
        for device in &self.devices {
            let target = device
                .compute_capability()
                .map(Target::arch)
                .with_context(|| format!("reading device {}'s architecture", device.0))?;
            if !targets.contains(&target) {
                targets.push(target);
            }
        }
        Ok(targets)
    }
}

/// The deployment's ranks the main process of `launch` holds, in rank order, each as its number in
/// the deployment, the index of its worker group there, its position in the group and the device
/// that places it: every rank of every worker group the deployment places on the process's host
/// (`EngineArgs::host`). A rank's number is its place among the ranks of the deployment's groups
/// in their order (`WorkerGroupConfig::first_ranks`), whatever order the hosts hold them in, so
/// the ranks of one host need not be consecutive; it is what the rank meets the deployment's host
/// group under and what its storage scope names. The one rule of a text and a media deployment,
/// and of the scopes the process claims at its host's Storage Server.
///
/// # Errors
///
/// A rank on this host placed on a device that is not a CUDA device the process serves
/// (`EngineArgs::devices`), or a rank past what the protocol numbers.
pub(crate) fn placed_ranks(launch: &EngineArgs) -> Result<Vec<(u32, usize, usize, DeviceId)>> {
    let groups = &launch.engine_config.parallel.worker_groups;
    let firsts = WorkerGroupConfig::first_ranks(groups);

    let mut placed = Vec::new();
    for (index, (first, group)) in firsts.iter().zip(groups).enumerate() {
        for (position, rank) in group.ranks.iter().enumerate() {
            if rank.host != launch.host {
                continue;
            }
            let number =
                u32::try_from(first + position).context("a rank past what the protocol numbers")?;
            let device = rank
                .device
                .strip_prefix("cuda:")
                .and_then(|ordinal| ordinal.parse::<u32>().ok())
                .filter(|ordinal| launch.devices.contains(ordinal))
                .and_then(|ordinal| i32::try_from(ordinal).ok());
            let Some(device) = device else {
                bail!(
                    "the deployment places rank {number} (rank {position} of worker group {}) on \
                     {}, which this process does not serve: it serves devices {:?}",
                    group.id,
                    rank.device,
                    launch.devices
                );
            };
            placed.push((number, index, position, DeviceId(device)));
        }
    }
    placed.sort_unstable_by_key(|(number, ..)| *number);
    Ok(placed)
}

/// The split a runner's graphs state for this rank, member `rank` of a parallel group of `ranks`
/// (`crate::runner::graphs::Graphs::set_group`): a group of one rank is no partition
/// (`nn::collective::Split::WHOLE`) and forms no communicator (`process::ranks::group_over`), so
/// its forwards state none, `(0, 0)`; a context that launches refuses a split without a
/// communicator.
pub(crate) fn split(rank: usize, ranks: usize) -> (u64, u64) {
    if ranks > 1 {
        (rank as u64, ranks as u64)
    } else {
        (0, 0)
    }
}

/// The tasks of host work one thread of the media pool has room for.
const MEDIA_TASKS: usize = 4;

/// The tasks the process's media pool holds at once: the requests a host phase's stage holds
/// (`Capacity::rows`).
pub(crate) fn media_tasks() -> usize {
    host_share().saturating_mul(MEDIA_TASKS)
}

/// The threads a pool of host work takes when nothing states its width: half the host's
/// parallelism, within one to eight.
fn host_share() -> usize {
    let available = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(4);
    (available / 2).clamp(1, 8)
}

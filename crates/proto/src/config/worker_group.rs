//! The worker groups of a deployment: which ranks each group has, on which
//! hosts and devices, and which blocks it runs on them.
//!
//! A worker is one GPU's execution thread in an engine process; a worker group
//! is the ranks that run the same blocks together and share a queue depth. The
//! deployment file states the groups, or the command-line shorthand builds one
//! group per scheduler ([`WorkerGroupConfig::data_parallel`]).

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::{BlockConfig, DeviceKind, RankDevice};

/// The block used when a deployment does not partition a model by capability.
pub const DEFAULT_BLOCK: &str = "model";

/// The block that decodes video latents into frames. Each member rank decodes
/// the temporal units dealt to it, one per round, so the block is always
/// distributed by temporal units and a deployment names no distribution for
/// it.
pub const VIDEO_DECODER_BLOCK: &str = "video_decoder";

/// The host block that encodes decoded frames, one codec slot per member rank.
/// It is always distributed by temporal units, one unit per rank, and a
/// deployment names no distribution for it.
pub const VIDEO_CODEC_BLOCK: &str = "video_codec";

/// Stable identity of one explicitly configured physical worker pool.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WorkerGroupId(pub String);

impl WorkerGroupId {
    /// Validates and constructs a stable pool identifier.
    ///
    /// An identifier is nonempty ASCII alphanumerics, `-`, and `_`. Only this
    /// constructor checks that; the public field admits any string.
    pub fn new(value: impl Into<String>) -> anyhow::Result<Self> {
        let value = value.into();
        if value.is_empty()
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        {
            anyhow::bail!("invalid worker id {value:?}");
        }
        Ok(Self(value))
    }
}

impl std::fmt::Display for WorkerGroupId {
    /// Formats the value for diagnostic output.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// One cooperative member's physical execution endpoint.
///
/// The deployment file writes a group's ranks compactly as `[HOST/]DEVICE` entries (`ranks`), and
/// a host the file leaves out is the engine's own, filled in when the file is read
/// ([`WorkerGroupConfig::parse_file`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerRank {
    /// Host identity: the daemon of the node with this host identity starts
    /// the rank.
    pub host: String,
    /// The rank's device on its host, such as `cuda:0` for a GPU rank or `cpu` for a CPU worker.
    pub device: RankDevice,
}

/// The numerical work a worker group owns in a deployment.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerGroupRole {
    /// Model capabilities, request state, and (when present) attention/KV.
    #[default]
    Model,
    /// A model group that prefills prompts only: a request continues on a
    /// `Decode` group of its scheduler, into whose device cache the scheduler
    /// copies the request's units (prefill/decode disaggregation).
    Prefill,
    /// A model group that decodes only, continuing requests a `Prefill`
    /// group prefilled.
    Decode,
    /// Routed experts shared by the deployment's schedulers.
    Experts,
}

impl WorkerGroupRole {
    /// Whether the group serves requests: every role but `Experts`.
    pub fn serves_requests(self) -> bool {
        self != Self::Experts
    }
}

/// The blocks a group of workers runs and their ordered physical rank membership: one
/// `[[worker_groups]]` table of the deployment file.
///
/// The deployment file ([`WorkerGroupConfig::parse_file`]) or the command-line shorthand
/// ([`WorkerGroupConfig::data_parallel`]) builds them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerGroupConfig {
    /// Group identity; replica `i` of a replicated group is `<id>-<i>`.
    pub id: WorkerGroupId,
    #[serde(default)]
    pub role: WorkerGroupRole,
    /// Copies of the group the file states, each over an equal consecutive block of `ranks`; the
    /// copies' blocks are replicas of the same numerical blocks under one scheduler, which binds
    /// each request to one copy. Reading the file expands them, so a read group has one.
    #[serde(default = "one", skip_serializing_if = "is_one")]
    pub replicas: usize,
    /// Ordered physical members; a rank's position is its rank number within
    /// the group.
    #[serde(with = "ranks")]
    pub ranks: Vec<WorkerRank>,
    /// Blocks keyed by name, each listing indices into `ranks`.
    #[serde(default)]
    pub blocks: BTreeMap<String, BlockConfig>,
    /// Maximum number of physical runs in flight per rank.
    pub queue_depth: usize,
    /// Per-rank share of device storage available to this group for static
    /// allocations (weights and KV cache), instead of the deployment-wide
    /// `--mem-fraction-static`.
    ///
    /// Distinct groups may share one physical GPU. Their configured
    /// shares must leave enough aggregate headroom for the device runtime;
    /// validation checks each group alone, never the sum across groups, and
    /// accepts the range [`WorkerGroupConfig::validate_mem_fraction_static`] states.
    #[serde(default)]
    pub mem_fraction_static: Option<f64>,
}

/// Serde default for a replica count.
const fn one() -> usize {
    1
}

fn is_one(count: &usize) -> bool {
    *count == 1
}

/// The deployment file: its `[[worker_groups]]` tables.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    worker_groups: Vec<WorkerGroupConfig>,
}

impl WorkerGroupConfig {
    /// The groups a deployment file states, with its ranks placed and its replicas expanded.
    ///
    /// A rank entry without a host runs on `engine_host`, the launching process's own host
    /// identity. A group with `replicas = N` becomes `N` groups named `<id>-0` through
    /// `<id>-<N-1>`, each over the next equal block of its ranks; a block without `ranks` spans
    /// every rank of its replica. The video decoder and the video codec always take one temporal
    /// unit per rank, so their distribution is completed and any other unit count refused.
    ///
    /// # Errors
    ///
    /// Fails when the text is not a deployment file, names no group, a group's replica count is
    /// zero or does not divide its ranks, the decoder or codec states another unit count, or
    /// [`WorkerGroupConfig::validate_all`] refuses the expanded groups.
    pub fn parse_file(text: &str, engine_host: &str) -> anyhow::Result<Vec<Self>> {
        use anyhow::Context as _;
        let file: File = toml::from_str(text)?;
        anyhow::ensure!(
            !file.worker_groups.is_empty(),
            "a deployment lists at least one [[worker_groups]] group"
        );
        let mut groups = Vec::new();
        for group in file.worker_groups {
            let id = group.id.clone();
            group
                .expand(engine_host, &mut groups)
                .with_context(|| format!("worker group {id}"))?;
        }
        Self::validate_all(&groups)?;
        Ok(groups)
    }

    /// Appends this file group's replicas to `groups`, placed on `engine_host`.
    fn expand(self, engine_host: &str, groups: &mut Vec<Self>) -> anyhow::Result<()> {
        use anyhow::Context as _;
        let ranks: Vec<WorkerRank> = self
            .ranks
            .iter()
            .map(|rank| WorkerRank {
                host: if rank.host.is_empty() {
                    engine_host.to_owned()
                } else {
                    rank.host.clone()
                },
                device: rank.device,
            })
            .collect();
        anyhow::ensure!(
            self.replicas > 0 && !ranks.is_empty() && ranks.len().is_multiple_of(self.replicas),
            "{} ranks do not form {} equal replicas",
            ranks.len(),
            self.replicas
        );
        let per_replica = ranks.len() / self.replicas;
        let mut blocks = self.blocks.clone();
        for (name, block) in &mut blocks {
            // The video decoder and codec run one local copy per member rank and take one
            // independent media unit per rank per round, the only layout a rank accepts for them.
            if [VIDEO_DECODER_BLOCK, VIDEO_CODEC_BLOCK].contains(&name.as_str()) {
                anyhow::ensure!(
                    block.units_per_rank == 1,
                    "block {name}: {name} takes one media unit per rank, not {}",
                    block.units_per_rank
                );
                block
                    .distribution
                    .get_or_insert(super::BlockDistribution::TemporalUnits);
            }
            if block.ranks.is_empty() {
                block.ranks = (0..per_replica).collect();
            }
        }
        for (index, replica) in ranks.chunks(per_replica).enumerate() {
            let id = if self.replicas == 1 {
                self.id.0.clone()
            } else {
                format!("{}-{index}", self.id.0)
            };
            groups.push(Self {
                id: WorkerGroupId::new(id.clone()).with_context(|| format!("group id {id}"))?,
                role: self.role,
                replicas: 1,
                ranks: replica.to_vec(),
                blocks: blocks.clone(),
                queue_depth: self.queue_depth,
                mem_fraction_static: self.mem_fraction_static,
            });
        }
        Ok(())
    }
}

/// A group's `ranks` as the deployment file writes them: `[HOST/]DEVICE` entries, each naming one
/// or more consecutive ranks. `DEVICE` is a single device (`cuda:0`, `cpu`), an inclusive index
/// range (`cuda:0-3`) or a repeated device (`cpu*4`); an entry without `HOST` leaves the host empty
/// for the reader to fill with the engine's.
mod ranks {
    use serde::{Deserialize, Deserializer, Serializer};

    use super::WorkerRank;

    pub(super) fn serialize<S: Serializer>(
        ranks: &[WorkerRank],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(ranks.iter().map(|rank| {
            if rank.host.is_empty() {
                rank.device.to_string()
            } else {
                format!("{}/{}", rank.host, rank.device)
            }
        }))
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vec<WorkerRank>, D::Error> {
        let entries = Vec::<String>::deserialize(deserializer)?;
        let mut ranks = Vec::new();
        for entry in entries {
            ranks.extend(expand(&entry).map_err(serde::de::Error::custom)?);
        }
        Ok(ranks)
    }

    fn expand(text: &str) -> Result<Vec<WorkerRank>, String> {
        let invalid = |reason: &str| format!("rank entry {text:?}: {reason}");
        let (host, device) = match text.split_once('/') {
            Some((host, device)) => (host, device),
            None => ("", text),
        };
        if text.contains('/') && host.is_empty() {
            return Err(invalid("the host before `/` is empty"));
        }
        let devices = if let Some((device, count)) = device.split_once('*') {
            let count = count
                .parse::<usize>()
                .ok()
                .filter(|count| *count > 0)
                .ok_or_else(|| invalid("the count after `*` must be a positive integer"))?;
            vec![device.to_owned(); count]
        } else if let Some((kind, indices)) = device.split_once(':')
            && let Some((first, last)) = indices.split_once('-')
        {
            let (Ok(first), Ok(last)) = (first.parse::<u32>(), last.parse::<u32>()) else {
                return Err(invalid("a range is `KIND:FIRST-LAST` with integer indices"));
            };
            if first > last {
                return Err(invalid("a range's first index exceeds its last"));
            }
            (first..=last)
                .map(|index| format!("{kind}:{index}"))
                .collect()
        } else {
            vec![device.to_owned()]
        };
        if devices.iter().any(String::is_empty) {
            return Err(invalid("the device is empty"));
        }
        devices
            .into_iter()
            .map(|device| {
                Ok(WorkerRank {
                    host: host.to_owned(),
                    device: device
                        .parse()
                        .map_err(|error: anyhow::Error| invalid(&error.to_string()))?,
                })
            })
            .collect()
    }
}

impl WorkerGroupConfig {
    /// Validates one group in isolation.
    ///
    /// Fails when the identity is not a valid `WorkerGroupId`, `queue_depth` is
    /// zero, `validate_mem_fraction_static` refuses a set `mem_fraction_static`, or
    /// `validate_members` rejects the ranks and blocks.
    pub fn validate(&self) -> anyhow::Result<()> {
        WorkerGroupId::new(self.id.0.clone())?;
        anyhow::ensure!(
            self.queue_depth > 0,
            "worker {} queue depth must be positive",
            self.id
        );
        if let Some(fraction) = self.mem_fraction_static {
            Self::validate_mem_fraction_static(fraction)
                .map_err(|error| anyhow::anyhow!("worker {}: {error}", self.id))?;
        }
        Self::validate_members(&self.ranks, &self.blocks)?;
        anyhow::ensure!(
            self.blocks.is_empty() != self.role.serves_requests(),
            "model workers require blocks; expert workers have no request blocks"
        );
        Ok(())
    }

    /// Checks a group's static storage fraction, `mem_fraction_static`.
    ///
    /// Each rank grants itself this share of every device's total storage,
    /// less what it already holds and never more than the device has free, so
    /// any finite value in (0, 1] is meaningful, 1 granting the whole device.
    pub fn validate_mem_fraction_static(fraction: f64) -> anyhow::Result<()> {
        anyhow::ensure!(
            fraction.is_finite() && fraction > 0.0 && fraction <= 1.0,
            "mem_fraction_static {fraction} must be finite and in (0, 1]"
        );
        Ok(())
    }

    /// Validates a group's rank list and its block membership.
    ///
    /// Ranks must be nonempty, name a host and a device, and not repeat a
    /// `(host, device)` pair unless the device is `cpu`. Blocks must be
    /// valid for their worker role; each needs a nonempty name, a nonempty list of distinct
    /// in-range rank indices, a `parallel_config` whose
    /// `BlockConfig::world_size` succeeds, and a positive `units_per_rank`.
    /// A block without a distribution must have a parallel world size
    /// equal to its rank count; one with a distribution must have world size
    /// one.
    pub fn validate_members(
        ranks: &[WorkerRank],
        blocks: &BTreeMap<String, BlockConfig>,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(!ranks.is_empty(), "worker requires rank members");
        let mut devices = BTreeSet::new();
        for rank in ranks {
            anyhow::ensure!(!rank.host.is_empty(), "rank requires a host");
            // Within a group a `(host, device)` pair names one rank, except
            // `cpu`, which several ranks on one host may share.
            anyhow::ensure!(
                devices.insert((&rank.host, rank.device)) || rank.device.is_shared(),
                "worker repeats a physical device"
            );
        }
        for (name, entry) in blocks {
            anyhow::ensure!(!name.is_empty(), "block name must not be empty");
            anyhow::ensure!(
                !entry.ranks.is_empty() && entry.ranks.iter().all(|&rank| rank < ranks.len()),
                "block {name} contains invalid members"
            );
            entry.validate()?;
        }
        Ok(())
    }

    /// Validates unique group identities and every local rank layout.
    ///
    /// Block names may repeat across groups. Repeated names are
    /// replicas of the same numerical block; the scheduler binds each
    /// request to one of them and keeps that affinity for the request lifetime.
    ///
    /// Physical devices may repeat across groups; only `validate_members`
    /// checks device reuse, and it does so within one group.
    pub fn validate_all(workers: &[Self]) -> anyhow::Result<()> {
        anyhow::ensure!(!workers.is_empty(), "engine requires workers");
        let mut ids = BTreeSet::new();
        for worker in workers {
            worker.validate()?;
            anyhow::ensure!(ids.insert(&worker.id), "worker group identity is repeated");
        }
        // A request prefilled on a prefill group decodes on a decode group,
        // which holds the same text model and nothing else.
        let has = |role| workers.iter().any(|worker| worker.role == role);
        anyhow::ensure!(
            has(WorkerGroupRole::Prefill) == has(WorkerGroupRole::Decode),
            "prefill groups and decode groups are configured together"
        );
        // A hand-off copies each prefill rank's units into the decode rank of the same index, so
        // every prefill and decode group has as many ranks as the others.
        let mut split_ranks: Option<(usize, &WorkerGroupId)> = None;
        for worker in workers {
            if matches!(
                worker.role,
                WorkerGroupRole::Prefill | WorkerGroupRole::Decode
            ) {
                anyhow::ensure!(
                    worker.blocks.keys().eq([DEFAULT_BLOCK]),
                    "worker {}: a prefill or decode group places the `{DEFAULT_BLOCK}` block alone",
                    worker.id
                );
                match split_ranks {
                    Some((ranks, first)) => anyhow::ensure!(
                        ranks == worker.ranks.len(),
                        "worker {} has {} ranks and worker {first} has {ranks}: prefill and decode \
                         groups hand a request's cache rank by rank",
                        worker.id,
                        worker.ranks.len()
                    ),
                    None => split_ranks = Some((worker.ranks.len(), &worker.id)),
                }
            }
        }
        Ok(())
    }

    /// The group that holds the deployment's rank `rank` and the rank's position in it.
    ///
    /// The deployment numbers its ranks over every group in the order the groups are stated, each
    /// group's ranks in their own order, as the daemon places one main process per host over that
    /// numbering; `None` past the last rank.
    pub fn place_rank(groups: &[Self], rank: usize) -> Option<(usize, usize)> {
        let mut first = 0;
        for (index, group) in groups.iter().enumerate() {
            if rank < first + group.ranks.len() {
                return Some((index, rank - first));
            }
            first += group.ranks.len();
        }
        None
    }

    /// The share of its GPU's storage a rank of this group may hold: the group's own
    /// `mem_fraction_static`, or `deployment`, the deployment-wide fraction, when the group
    /// states none.
    pub fn fraction(&self, deployment: f64) -> f64 {
        self.mem_fraction_static.unwrap_or(deployment)
    }

    /// The fraction of the GPU at `placed` (its host and device) that the deployment's quota of
    /// the GPU is limited to: the shares of every rank `groups` place there, where a rank whose
    /// group states its own `mem_fraction_static` adds that share and the ranks that take the
    /// deployment's share hold `deployment` together; never more than the whole GPU.
    ///
    /// One deployment has one quota per GPU, whose limit an epoch states once
    /// (`docs/admin.md`), so every rank on the GPU states this one figure, and each rank's own
    /// share ([`Self::fraction`]) bounds what the rank itself divides.
    pub fn device_fraction(groups: &[Self], placed: &WorkerRank, deployment: f64) -> f64 {
        let mut own = 0.0;
        let mut shares_deployment = false;
        for group in groups {
            let ranks = group.ranks.iter().filter(|rank| *rank == placed).count();
            match group.mem_fraction_static {
                Some(fraction) => own += fraction * ranks as f64,
                None => shares_deployment |= ranks > 0,
            }
        }

        let deployment = if shares_deployment { deployment } else { 0.0 };
        (own + deployment).min(1.0)
    }

    /// The hosts `groups` place ranks on, each once, in the order of the first rank each holds,
    /// with that rank's deployment rank: the hosts a deployment file names, as its reader placed
    /// them.
    pub fn hosts(groups: &[Self]) -> Vec<(&str, usize)> {
        let ranks = groups.iter().flat_map(|group| group.ranks.iter());
        let mut firsts: Vec<(&str, usize)> = Vec::new();
        for (rank, placed) in ranks.enumerate() {
            if !firsts.iter().any(|(host, _)| *host == placed.host) {
                firsts.push((placed.host.as_str(), rank));
            }
        }
        firsts
    }

    /// The deployment rank of position 0 of each group, in the order the groups are stated.
    pub fn first_ranks(groups: &[Self]) -> Vec<usize> {
        groups
            .iter()
            .scan(0, |first, group| {
                let at = *first;
                *first += group.ranks.len();
                Some(at)
            })
            .collect()
    }

    /// The groups each of `data_parallel` schedulers serves, by index into `groups`: the groups
    /// that serve requests, in order, dealt as equal consecutive blocks.
    ///
    /// A scheduler drives every group of its block as one of its worker groups, so a deployment's
    /// prefill and decode groups are one scheduler's, and the shorthand's `model-<i>` group is
    /// scheduler `i`'s alone.
    ///
    /// # Errors
    ///
    /// Fails when the serving groups do not divide into `data_parallel` equal blocks.
    pub fn scheduler_blocks(
        groups: &[Self],
        data_parallel: usize,
    ) -> anyhow::Result<Vec<Vec<usize>>> {
        let serving: Vec<usize> = groups
            .iter()
            .enumerate()
            .filter(|(_, group)| group.role.serves_requests())
            .map(|(index, _)| index)
            .collect();
        let data_parallel = data_parallel.max(1);
        anyhow::ensure!(
            !serving.is_empty() && serving.len().is_multiple_of(data_parallel),
            "{} groups serve requests, which do not form {data_parallel} equal blocks of \
             data-parallel schedulers",
            serving.len()
        );
        Ok(serving
            .chunks(serving.len() / data_parallel)
            .map(<[usize]>::to_vec)
            .collect())
    }

    /// One block tensor-parallel over every rank.
    ///
    /// The `uniserve` CLI uses this rank layout when no `--worker-groups` configuration
    /// is given, and the default engine settings and the simulation use it. A
    /// deployment with several blocks, or with other partitions, writes
    /// its own configuration.
    pub fn single_block(name: &str, rank_count: usize) -> BTreeMap<String, BlockConfig> {
        BTreeMap::from([(
            name.into(),
            BlockConfig {
                tensor_parallel_size: rank_count,
                ..BlockConfig::parallel((0..rank_count).collect())
            },
        )])
    }

    /// Expands the single-group CLI shorthand into explicit device membership
    /// across the named hosts, carrying the given blocks.
    ///
    /// Ranks are assigned in blocks, so the lowest ranks stay on the first
    /// host, and each host numbers its devices from zero. A rank layout on one
    /// host is the same arithmetic with one block. Each rank's device is
    /// `device`'s for its index on its host ([`DeviceKind::rank_device`]).
    ///
    /// `blocks` must already be resolved for `rank_count`; this assigns
    /// rank identity and nothing else. The group is named after the shorthand's
    /// block, [`DEFAULT_BLOCK`], and has no `mem_fraction_static` of its own.
    /// Nothing is validated here, and an empty `hosts` yields no ranks.
    pub fn placed(
        hosts: &[String],
        device: DeviceKind,
        rank_count: usize,
        queue_depth: usize,
        blocks: BTreeMap<String, BlockConfig>,
    ) -> Self {
        // A host takes its share of the ranks, and the first hosts take the
        // remainder, so a count that does not divide evenly still places every
        // rank and keeps each host's block contiguous.
        let host_count = hosts.len().max(1);
        let share = rank_count / host_count;
        let remainder = rank_count % host_count;
        let mut rank_layout = Vec::with_capacity(rank_count);
        for (index, host) in hosts.iter().enumerate() {
            let count = share + usize::from(index < remainder);
            for local in 0..count {
                rank_layout.push(WorkerRank {
                    host: host.clone(),
                    device: device.rank_device(local as u32),
                });
            }
        }
        Self {
            id: WorkerGroupId(DEFAULT_BLOCK.into()),
            role: WorkerGroupRole::Model,
            replicas: 1,
            ranks: rank_layout,
            blocks,
            queue_depth,
            mem_fraction_static: None,
        }
    }

    /// Expands the data-parallel CLI shorthand into one group of
    /// `ranks_per_core` ranks for each of `cores` schedulers, each carrying
    /// `blocks`.
    ///
    /// All `cores * ranks_per_core` ranks are placed at once as
    /// [`WorkerGroupConfig::placed`] places them, in blocks over `hosts`, and
    /// consecutive runs of `ranks_per_core` form the groups, so the first
    /// schedulers fill the engine's host first. The group of one scheduler
    /// is named after the shorthand's block, [`DEFAULT_BLOCK`]; the group of
    /// core `i` of several is `<block>-<i>`.
    pub fn data_parallel(
        hosts: &[String],
        device: DeviceKind,
        cores: usize,
        ranks_per_core: usize,
        queue_depth: usize,
        blocks: BTreeMap<String, BlockConfig>,
    ) -> Vec<Self> {
        let placed = Self::placed(hosts, device, cores * ranks_per_core, queue_depth, blocks);
        if cores <= 1 {
            return vec![placed];
        }
        placed
            .ranks
            .chunks(ranks_per_core.max(1))
            .enumerate()
            .map(|(index, ranks)| Self {
                id: WorkerGroupId(format!("{DEFAULT_BLOCK}-{index}")),
                role: WorkerGroupRole::Model,
                replicas: 1,
                ranks: ranks.to_vec(),
                blocks: placed.blocks.clone(),
                queue_depth,
                mem_fraction_static: None,
            })
            .collect()
    }
}

impl WorkerGroupConfig {
    /// The command line's shorthand: `data_parallel_size` groups of `tensor_parallel_size` ranks,
    /// each running the default block over all its ranks, spread in blocks over `engine_host` and
    /// the hosts after it (`device_config.worker_hosts`), on the devices `device` states.
    ///
    /// # Errors
    ///
    /// Fails when `device.devices` orders the ranks but is not one CUDA device per rank, or puts two
    /// ranks of a host on one device.
    pub fn shorthand(
        engine_host: &str,
        device: &super::DeviceConfig,
        tensor_parallel_size: usize,
        data_parallel_size: usize,
    ) -> anyhow::Result<Vec<Self>> {
        // Leading with the leader's own host keeps rank zero and the other lowest ranks there; a
        // repeat of it, or an empty name, places nothing.
        let mut hosts = vec![engine_host.to_owned()];
        hosts.extend(
            device
                .worker_hosts
                .iter()
                .filter(|host| !host.is_empty() && *host != engine_host)
                .cloned(),
        );
        let tensor_parallel_size = tensor_parallel_size.max(1);
        let mut groups = Self::data_parallel(
            &hosts,
            device.device,
            data_parallel_size.max(1),
            tensor_parallel_size,
            device.queue_depth.get(),
            Self::single_block(DEFAULT_BLOCK, tensor_parallel_size),
        );
        if !device.devices.is_empty() {
            order_devices(&mut groups, device.device, &device.devices)?;
        }
        Ok(groups)
    }
}

/// Puts the shorthand's ranks, in rank order across `groups`, on the CUDA devices `devices` names
/// by their index on each rank's host (`device_config.devices`): an epoch whose ranks a daemon
/// assigned to the GPUs that already hold most of their weights, or an operator's order.
fn order_devices(
    groups: &mut [WorkerGroupConfig],
    device: DeviceKind,
    devices: &[u32],
) -> anyhow::Result<()> {
    anyhow::ensure!(
        device == DeviceKind::Cuda,
        "`device_config.devices` orders CUDA devices; `device_config.device` is `{device}`"
    );
    let ranks: usize = groups.iter().map(|group| group.ranks.len()).sum();
    anyhow::ensure!(
        devices.len() == ranks,
        "`device_config.devices` names {} devices for {ranks} ranks",
        devices.len()
    );
    let mut taken = BTreeSet::new();
    let ranks = groups.iter_mut().flat_map(|group| group.ranks.iter_mut());
    for (rank, index) in ranks.zip(devices) {
        anyhow::ensure!(
            taken.insert((rank.host.clone(), *index)),
            "`device_config.devices` puts two ranks of host {} on device {index}",
            rank.host
        );
        rank.device = DeviceKind::Cuda.rank_device(*index);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BlockDistribution;

    fn workers(text: &str) -> anyhow::Result<Vec<WorkerGroupConfig>> {
        WorkerGroupConfig::parse_file(text, "engine")
    }

    fn rank(host: &str, device: &str) -> WorkerRank {
        WorkerRank {
            host: host.into(),
            device: device.parse().unwrap(),
        }
    }

    /// The command-line shorthand's groups for `tensor` tensor-parallel ranks per scheduler and
    /// `data` schedulers on this host, on the devices `devices` orders.
    fn shorthand(
        tensor: usize,
        data: usize,
        devices: &[u32],
    ) -> anyhow::Result<Vec<WorkerGroupConfig>> {
        let device = crate::config::DeviceConfig {
            devices: devices.to_vec(),
            ..crate::config::DeviceConfig::default()
        };
        WorkerGroupConfig::shorthand("localhost", &device, tensor, data)
    }

    /// The devices of `groups`, group by group in rank order.
    fn devices(groups: &[WorkerGroupConfig]) -> Vec<(String, Vec<String>)> {
        groups
            .iter()
            .map(|group| {
                (
                    group.id.to_string(),
                    group
                        .ranks
                        .iter()
                        .map(|rank| rank.device.to_string())
                        .collect(),
                )
            })
            .collect()
    }

    /// The shorthand puts its ranks, in rank order, on the devices
    /// `device_config.devices` names.
    #[test]
    fn the_shorthand_puts_ranks_on_the_devices_their_order_names() {
        let ordered = shorthand(4, 1, &[0, 2, 1, 3]).unwrap();
        assert_eq!(
            devices(&ordered),
            [(
                "model".to_owned(),
                vec![
                    "cuda:0".to_owned(),
                    "cuda:2".to_owned(),
                    "cuda:1".to_owned(),
                    "cuda:3".to_owned(),
                ],
            )]
        );

        // Without an order each host's ranks number its devices from zero.
        let numbered = shorthand(4, 1, &[]).unwrap();
        assert_eq!(
            devices(&numbered),
            [(
                "model".to_owned(),
                vec![
                    "cuda:0".to_owned(),
                    "cuda:1".to_owned(),
                    "cuda:2".to_owned(),
                    "cuda:3".to_owned(),
                ],
            )]
        );
    }

    /// The order runs across the schedulers' groups: the devices name each scheduler's rank in
    /// turn.
    #[test]
    fn a_device_order_runs_across_the_schedulers_groups() {
        let groups = shorthand(1, 2, &[1, 0]).unwrap();
        assert_eq!(
            devices(&groups),
            [
                ("model-0".to_owned(), vec!["cuda:1".to_owned()]),
                ("model-1".to_owned(), vec!["cuda:0".to_owned()]),
            ]
        );
    }

    /// An order that does not name one CUDA device per rank, or puts two ranks of a host on one
    /// device, is refused naming the option, and so is an order of devices that are not CUDA's.
    #[test]
    fn a_device_order_of_another_length_or_a_repeated_device_is_refused() {
        for order in [&[0, 1, 2][..], &[0, 1, 1, 2]] {
            let error = shorthand(4, 1, order).expect_err("the order must be refused");
            assert!(
                format!("{error:#}").contains("device_config.devices"),
                "{error:#}"
            );
        }
        let cpu = crate::config::DeviceConfig {
            device: DeviceKind::Cpu,
            devices: vec![0, 1],
            ..crate::config::DeviceConfig::default()
        };
        assert!(WorkerGroupConfig::shorthand("localhost", &cpu, 2, 1).is_err());
    }

    /// Replicas take equal consecutive blocks of the stated ranks and are
    /// named by their index; blocks without ranks span each replica.
    #[test]
    fn replicas_split_the_ranks_into_named_blocks() {
        let placed = workers(
            r#"
            [[worker_groups]]
            id = "flow"
            replicas = 2
            ranks = ["cuda:0-3"]
            queue_depth = 6
            mem_fraction_static = 0.5

            [worker_groups.blocks.transformer]
            sequence_parallel = { kind = "ulysses", ulysses_degree = 2 }
            "#,
        )
        .unwrap();

        assert_eq!(placed.len(), 2);
        assert_eq!(placed[0].id.0, "flow-0");
        assert_eq!(placed[1].id.0, "flow-1");
        assert_eq!(
            placed[1].ranks,
            vec![rank("engine", "cuda:2"), rank("engine", "cuda:3")]
        );
        for worker in &placed {
            assert_eq!(worker.queue_depth, 6);
            assert_eq!(worker.mem_fraction_static, Some(0.5));
            let transformer = &worker.blocks["transformer"];
            assert_eq!(transformer.ranks, vec![0, 1]);
            assert_eq!(transformer.world_size().unwrap(), 2);
        }
    }

    /// An entry without a host runs on the engine's host; a named host, a
    /// repeated host device and a single device each place as written.
    #[test]
    fn rank_entries_place_on_the_engines_host_or_a_named_host() {
        let placed = workers(
            r#"
            [[worker_groups]]
            id = "media"
            ranks = ["cpu*2", "rank-1/cpu", "rank-1/cuda:1"]
            queue_depth = 2

            [worker_groups.blocks.muxer]
            ranks = [0]
            "#,
        )
        .unwrap();

        assert_eq!(placed[0].id.0, "media");
        assert_eq!(
            placed[0].ranks,
            vec![
                rank("engine", "cpu"),
                rank("engine", "cpu"),
                rank("rank-1", "cpu"),
                rank("rank-1", "cuda:1"),
            ]
        );
    }

    /// The video decoder and codec are temporal-unit blocks whether or
    /// not the file says so, and no other unit count is a layout they
    /// have.
    #[test]
    fn the_video_decoder_and_codec_take_one_unit_per_rank() {
        let decoder = |table: &str| {
            workers(&format!(
                "[[worker_groups]]\nid = \"model\"\nranks = [\"cuda:0-1\"]\nqueue_depth = 2\n\
                 [worker_groups.blocks.video_decoder]\n{table}"
            ))
        };

        let omitted = decoder("").unwrap();
        let stated = decoder("distribution = \"temporal_units\"").unwrap();
        assert_eq!(omitted, stated);
        assert_eq!(
            omitted[0].blocks["video_decoder"].distribution,
            Some(BlockDistribution::TemporalUnits)
        );
        assert!(decoder("units_per_rank = 2").is_err());
    }

    #[test]
    fn malformed_deployments_are_refused() {
        let group =
            |body: &str| format!("[[worker_groups]]\nid = \"model\"\nqueue_depth = 2\n{body}");
        for refused in [
            // Unknown keys, at the group and the block.
            group("ranks = [\"cuda:0\"]\nreplica = 2"),
            group("ranks = [\"cuda:0\"]\n[worker_groups.blocks.model]\ntensor_parallel = 1"),
            // Rank entries that name no devices.
            group("ranks = [\"cuda:3-1\"]"),
            group("ranks = [\"cpu*0\"]"),
            group("ranks = [\"/cuda:0\"]"),
            // Replicas that do not divide the ranks.
            group("ranks = [\"cuda:0-2\"]\nreplicas = 2"),
            // A block whose degrees do not match its ranks.
            group("ranks = [\"cuda:0-1\"]\n[worker_groups.blocks.model]\ntensor_parallel_size = 4"),
            // No groups at all.
            String::new(),
        ] {
            assert!(workers(&refused).is_err(), "{refused:?} must be refused");
        }
    }

    /// Every deployment the repository publishes for `--worker-groups` places and validates with
    /// each host it names as the engine's, as the node of that host reads it, and places no rank
    /// on a CPU, which a node refuses: a main process serves CUDA devices.
    #[test]
    fn published_deployments_place() {
        let root = std::path::Path::new(crate::WORKSPACE).join("configs");
        let mut files = 0;
        for model in std::fs::read_dir(&root).unwrap() {
            for entry in std::fs::read_dir(model.unwrap().path()).unwrap() {
                let path = entry.unwrap().path();
                let text = std::fs::read_to_string(&path).unwrap();
                let place = |host: &str| {
                    WorkerGroupConfig::parse_file(&text, host).unwrap_or_else(|error| {
                        panic!("{} does not place on {host}: {error:#}", path.display())
                    })
                };
                let local = place("localhost");
                for (host, _) in WorkerGroupConfig::hosts(&local) {
                    for group in place(host) {
                        for rank in &group.ranks {
                            assert!(
                                rank.device.kind != DeviceKind::Cpu,
                                "{}: worker group {} places rank {}/{} on a CPU",
                                path.display(),
                                group.id,
                                rank.host,
                                rank.device
                            );
                        }
                    }
                }
                files += 1;
            }
        }
        assert!(files > 0, "configs/ publishes deployments");
    }

    /// A prefill/decode deployment's groups are one scheduler's, the shorthand's data-parallel
    /// groups one scheduler's each; a rank is found in its group by the deployment's numbering, and
    /// a prefill and a decode group of different rank counts are refused.
    #[test]
    fn serving_groups_form_the_schedulers_and_ranks_find_their_group() {
        let split = workers(
            "[[worker_groups]]\nid = \"prefill\"\nrole = \"prefill\"\nranks = [\"cuda:0\"]\n\
             queue_depth = 2\n[worker_groups.blocks.model]\n\
             [[worker_groups]]\nid = \"decode\"\nrole = \"decode\"\nranks = [\"r2/cuda:0\"]\n\
             queue_depth = 2\n[worker_groups.blocks.model]\n",
        )
        .unwrap();
        assert_eq!(
            WorkerGroupConfig::scheduler_blocks(&split, 1).unwrap(),
            vec![vec![0, 1]]
        );
        assert_eq!(WorkerGroupConfig::place_rank(&split, 1), Some((1, 0)));
        assert_eq!(WorkerGroupConfig::place_rank(&split, 2), None);

        let shorthand = WorkerGroupConfig::data_parallel(
            &["engine".to_owned()],
            DeviceKind::Cuda,
            2,
            2,
            2,
            WorkerGroupConfig::single_block(DEFAULT_BLOCK, 2),
        );
        assert_eq!(
            WorkerGroupConfig::scheduler_blocks(&shorthand, 2).unwrap(),
            vec![vec![0], vec![1]]
        );
        assert_eq!(WorkerGroupConfig::first_ranks(&shorthand), vec![0, 2]);
        assert_eq!(WorkerGroupConfig::place_rank(&shorthand, 3), Some((1, 1)));
        assert!(WorkerGroupConfig::scheduler_blocks(&shorthand, 3).is_err());

        let uneven = workers(
            "[[worker_groups]]\nid = \"prefill\"\nrole = \"prefill\"\nranks = [\"cuda:0-1\"]\n\
             queue_depth = 2\n[worker_groups.blocks.model]\ntensor_parallel_size = 2\n\
             [[worker_groups]]\nid = \"decode\"\nrole = \"decode\"\nranks = [\"cuda:2\"]\n\
             queue_depth = 2\n[worker_groups.blocks.model]\n",
        );
        assert!(uneven.is_err(), "a hand-off copies between equal ranks");
    }

    /// Each rank holds its group's share of its GPU; the deployment's quota of a GPU holds every
    /// rank's there, the ranks that take the deployment's share holding it together.
    #[test]
    fn a_gpu_s_quota_holds_the_shares_of_the_ranks_placed_on_it() {
        // A text encoder of 0.18 and a denoiser replica of 0.81 on each GPU.
        let colocated = workers(
            r#"
            [[worker_groups]]
            id = "text"
            ranks = ["cuda:0-1"]
            queue_depth = 2
            mem_fraction_static = 0.18
            [worker_groups.blocks.text_encoder]
            tensor_parallel_size = 2

            [[worker_groups]]
            id = "flow"
            replicas = 2
            ranks = ["cuda:0-1"]
            queue_depth = 2
            mem_fraction_static = 0.81
            [worker_groups.blocks.transformer]
            "#,
        )
        .unwrap();
        assert_eq!(colocated[0].fraction(0.7), 0.18);
        assert_eq!(colocated[1].fraction(0.7), 0.81);
        let gpu = WorkerGroupConfig::device_fraction(&colocated, &rank("engine", "cuda:1"), 0.7);
        assert!((gpu - 0.99).abs() < 1e-12, "{gpu}");

        // Ranks of groups that state no share hold the deployment's together, beside a group
        // that states its own; a device no rank is placed on holds nothing.
        let shared = workers(
            r#"
            [[worker_groups]]
            id = "model"
            ranks = ["cuda:0-1"]
            queue_depth = 2
            [worker_groups.blocks.transformer]
            sequence_parallel = { kind = "ulysses", ulysses_degree = 2 }

            [[worker_groups]]
            id = "muxer"
            ranks = ["cuda:0"]
            queue_depth = 2
            [worker_groups.blocks.muxer]

            [[worker_groups]]
            id = "reader"
            ranks = ["cuda:0"]
            queue_depth = 2
            mem_fraction_static = 0.04
            [worker_groups.blocks.media_reader]
            "#,
        )
        .unwrap();
        assert_eq!(shared[1].fraction(0.7), 0.7);
        let first = WorkerGroupConfig::device_fraction(&shared, &rank("engine", "cuda:0"), 0.7);
        assert!((first - 0.74).abs() < 1e-12, "{first}");
        let second = WorkerGroupConfig::device_fraction(&shared, &rank("engine", "cuda:1"), 0.7);
        assert_eq!(second, 0.7);
        let other = WorkerGroupConfig::device_fraction(&shared, &rank("other", "cuda:0"), 0.7);
        assert_eq!(other, 0.0);

        // The shares never take more than the whole GPU.
        let whole = WorkerGroupConfig::device_fraction(&colocated, &rank("engine", "cuda:0"), 0.7);
        assert!(whole <= 1.0);
        let mut doubled = colocated.clone();
        doubled[0].mem_fraction_static = Some(0.5);
        doubled[1].mem_fraction_static = Some(0.6);
        assert_eq!(
            WorkerGroupConfig::device_fraction(&doubled, &rank("engine", "cuda:0"), 0.7),
            1.0
        );
    }
}

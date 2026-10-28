//! Deployment files: the worker groups a server launches, stated in TOML.
//!
//! A deployment file lists worker groups as `[[worker_groups]]` tables. Each group
//! names its ranks compactly and the components placed on them:
//!
//! ```toml
//! [[worker_groups]]
//! id = "flow"
//! replicas = 2              # flow-0 on cuda:0-3, flow-1 on cuda:4-7
//! ranks = ["cuda:0-7"]
//! queue_depth = 6
//!
//! [worker_groups.components.transformer]
//! sequence_parallel = { kind = "ulysses", ulysses_degree = 4 }
//!
//! [worker_groups.components.audio_decoder]
//! distribution = "temporal_units"
//! ```
//!
//! [`Deployment::read`] parses a file; [`Deployment::worker_groups`] places its
//! ranks on hosts and expands it into the engine's [`WorkerGroupConfig`] list,
//! which every launch path consumes. Reading checks the file's syntax and
//! schema; placement validates the expanded groups with
//! [`WorkerGroupConfig::validate_all`], so a file that names no host can be
//! checked only once the engine's own host identity is known.

use std::collections::BTreeMap;

use anyhow::Context as _;
use ipc::{
    ComponentConfig, ComponentDistribution, ComponentWeights, ParallelConfig, SequenceParallel,
};
use ipc::{VIDEO_CODEC_COMPONENT, VIDEO_DECODER_COMPONENT};
use serde::{Deserialize, Serialize};

use crate::core::{WorkerGroupConfig, WorkerGroupRole, WorkerRank};
use crate::executor::WorkerGroupId;

/// Serde default for a degree, a replica count and `units_per_rank`.
const fn one() -> usize {
    1
}

/// A deployment as its file states it, before its ranks are placed on hosts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Deployment {
    /// Worker groups in file order; a replicated group expands in place.
    worker_groups: Vec<GroupEntry>,
}

/// One `[[worker_groups]]` table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GroupEntry {
    /// Group identity; replica `i` of a replicated group is `<id>-<i>`.
    id: String,
    #[serde(default)]
    role: WorkerGroupRole,
    /// Copies of the group, each over an equal consecutive block of `ranks`.
    /// The copies' components are replicas of the same numerical components
    /// under one scheduler, which binds each request to one copy.
    #[serde(default = "one")]
    replicas: usize,
    /// The group's ranks, every replica's together, in rank order.
    ranks: Vec<RankRange>,
    /// Physical runs in flight per rank (`WorkerProcessArgs::queue_depth`).
    queue_depth: usize,
    /// Per-rank device-storage share; the server-wide `mem_fraction_static`
    /// applies when absent.
    #[serde(default)]
    mem_fraction_static: Option<f64>,
    #[serde(default)]
    components: BTreeMap<String, ComponentEntry>,
}

/// One `[worker_groups.components.<name>]` table: a [`ComponentConfig`] whose
/// parallel degrees are stated beside its membership.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ComponentEntry {
    /// Indices into one replica's ranks; every rank of the replica when
    /// absent.
    #[serde(default)]
    ranks: Option<Vec<usize>>,
    #[serde(default = "one")]
    tensor_parallel_size: usize,
    #[serde(default = "one")]
    pipeline_parallel_size: usize,
    #[serde(default)]
    sequence_parallel: SequenceParallel,
    #[serde(default)]
    distribution: Option<ComponentDistribution>,
    #[serde(default = "one")]
    units_per_rank: usize,
    #[serde(default)]
    weights: ComponentWeights,
}

/// One `ranks` entry, `[HOST/]DEVICE`, naming one or more consecutive ranks.
///
/// `DEVICE` is a single device (`cuda:0`, `cpu`), an inclusive index range
/// (`cuda:0-3`) or a repeated device (`cpu*4`). Without `HOST` the ranks run
/// on the engine's own host.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(try_from = "String")]
struct RankRange {
    host: Option<String>,
    devices: Vec<String>,
}

impl Serialize for RankRange {
    #[expect(
        clippy::unwrap_used,
        reason = "parsing creates nonempty repeated devices or an indexed range"
    )]
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let first = &self.devices[0];
        let device = if self.devices.len() == 1 {
            first.clone()
        } else if self.devices.iter().all(|device| device == first) {
            format!("{first}*{}", self.devices.len())
        } else {
            // Parsing constructs either repeated devices or a consecutive
            // indexed range; snapshots use single-device entries.
            let (_, last) = self.devices.last().unwrap().split_once(':').unwrap();
            format!("{first}-{last}")
        };
        let rank = self
            .host
            .as_ref()
            .map_or_else(|| device.clone(), |host| format!("{host}/{device}"));
        serializer.serialize_str(&rank)
    }
}

impl TryFrom<String> for RankRange {
    type Error = String;

    fn try_from(text: String) -> Result<Self, Self::Error> {
        let invalid = |reason: &str| format!("rank entry {text:?}: {reason}");

        let (host, device) = match text.split_once('/') {
            Some((host, device)) => (Some(host.to_owned()), device),
            None => (None, text.as_str()),
        };
        if host.as_deref().is_some_and(str::is_empty) {
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

        Ok(Self { host, devices })
    }
}

impl Deployment {
    /// Encodes resolved worker groups as a deployment snapshot. Each rank
    /// names its host and device explicitly; component membership and group
    /// identities are preserved across independently launched replicas.
    pub fn snapshot(groups: &[WorkerGroupConfig]) -> anyhow::Result<String> {
        let deployment = Self {
            worker_groups: groups
                .iter()
                .map(|group| GroupEntry {
                    id: group.id.0.clone(),
                    role: group.role,
                    replicas: 1,
                    ranks: group
                        .ranks
                        .iter()
                        .map(|rank| RankRange {
                            host: Some(rank.host.clone()),
                            devices: vec![rank.device.clone()],
                        })
                        .collect(),
                    queue_depth: group.queue_depth,
                    mem_fraction_static: group.mem_fraction_static,
                    components: group
                        .components
                        .iter()
                        .map(|(name, component)| {
                            (
                                name.clone(),
                                ComponentEntry {
                                    ranks: Some(component.ranks.clone()),
                                    tensor_parallel_size: component
                                        .parallel_config
                                        .tensor_parallel_size,
                                    pipeline_parallel_size: component
                                        .parallel_config
                                        .pipeline_parallel_size,
                                    sequence_parallel: component
                                        .parallel_config
                                        .sequence_parallel
                                        .clone(),
                                    distribution: component.distribution,
                                    units_per_rank: component.units_per_rank,
                                    weights: component.weights,
                                },
                            )
                        })
                        .collect(),
                })
                .collect(),
        };
        Ok(toml::to_string(&deployment)?)
    }

    /// Reads and parses the deployment file at `path`.
    ///
    /// The signature suits a command-line value parser; each failure names
    /// the file and its cause in one message.
    pub fn read(path: &str) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading deployment file {path}"))?;
        Self::parse(&text).with_context(|| format!("parsing deployment file {path}"))
    }

    /// Parses deployment TOML text, refusing unknown keys and malformed
    /// rank entries.
    pub fn parse(text: &str) -> anyhow::Result<Self> {
        let deployment: Self = toml::from_str(text)?;
        anyhow::ensure!(
            !deployment.worker_groups.is_empty(),
            "a deployment lists at least one [[worker_groups]] group"
        );
        Ok(deployment)
    }

    /// Places the deployment's ranks and returns its worker groups.
    ///
    /// A rank entry without a host runs on `engine_host`, the launching
    /// process's own host identity. A group with `replicas = N` becomes `N`
    /// groups named `<id>-0` through `<id>-<N-1>`, each over the next equal
    /// block of its ranks; a component without `ranks` spans every rank of its
    /// replica. The video decoder and the video codec always take one
    /// temporal unit per rank, so their distribution is completed and any
    /// other unit count refused.
    ///
    /// # Errors
    ///
    /// Fails when a group's replica count is zero or does not divide its
    /// ranks, when the decoder or codec states another unit count, or when
    /// [`WorkerGroupConfig::validate_all`] refuses the expanded groups.
    pub fn worker_groups(&self, engine_host: &str) -> anyhow::Result<Vec<WorkerGroupConfig>> {
        let mut workers = Vec::new();
        for entry in &self.worker_groups {
            entry
                .expand(engine_host, &mut workers)
                .with_context(|| format!("worker {}", entry.id))?;
        }
        WorkerGroupConfig::validate_all(&workers)?;
        Ok(workers)
    }
}

impl GroupEntry {
    /// Appends this group's replicas to `workers`.
    fn expand(
        &self,
        engine_host: &str,
        workers: &mut Vec<WorkerGroupConfig>,
    ) -> anyhow::Result<()> {
        let ranks = self
            .ranks
            .iter()
            .flat_map(|range| {
                let host = range.host.as_deref().unwrap_or(engine_host);
                range.devices.iter().map(move |device| WorkerRank {
                    host: host.to_owned(),
                    device: device.clone(),
                })
            })
            .collect::<Vec<_>>();
        anyhow::ensure!(
            self.replicas > 0 && !ranks.is_empty() && ranks.len().is_multiple_of(self.replicas),
            "{} ranks do not form {} equal replicas",
            ranks.len(),
            self.replicas
        );
        let per_replica = ranks.len() / self.replicas;

        let components = self
            .components
            .iter()
            .map(|(name, component)| {
                let config = component
                    .resolve(name, per_replica)
                    .with_context(|| format!("component {name}"))?;
                Ok((name.clone(), config))
            })
            .collect::<anyhow::Result<BTreeMap<_, _>>>()?;

        for (index, block) in ranks.chunks(per_replica).enumerate() {
            let id = if self.replicas == 1 {
                self.id.clone()
            } else {
                format!("{}-{index}", self.id)
            };
            workers.push(WorkerGroupConfig {
                id: WorkerGroupId(id),
                role: self.role,
                ranks: block.to_vec(),
                components: components.clone(),
                queue_depth: self.queue_depth,
                mem_fraction_static: self.mem_fraction_static,
            });
        }
        Ok(())
    }
}

impl ComponentEntry {
    /// The component's placement within one replica of `rank_count` ranks.
    fn resolve(&self, name: &str, rank_count: usize) -> anyhow::Result<ComponentConfig> {
        // The video decoder and codec run one local copy per member rank
        // and take one independent media unit per rank per round, the only
        // placement a rank accepts for them.
        let mut distribution = self.distribution;
        if [VIDEO_DECODER_COMPONENT, VIDEO_CODEC_COMPONENT].contains(&name) {
            anyhow::ensure!(
                self.units_per_rank == 1,
                "{name} takes one media unit per rank, not {}",
                self.units_per_rank
            );
            distribution.get_or_insert(ComponentDistribution::TemporalUnits);
        }

        Ok(ComponentConfig {
            ranks: self
                .ranks
                .clone()
                .unwrap_or_else(|| (0..rank_count).collect()),
            parallel_config: ParallelConfig {
                tensor_parallel_size: self.tensor_parallel_size,
                pipeline_parallel_size: self.pipeline_parallel_size,
                sequence_parallel: self.sequence_parallel.clone(),
            },
            units_per_rank: self.units_per_rank,
            weights: self.weights,
            distribution,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workers(text: &str) -> anyhow::Result<Vec<WorkerGroupConfig>> {
        Deployment::parse(text)?.worker_groups("engine")
    }

    fn rank(host: &str, device: &str) -> WorkerRank {
        WorkerRank {
            host: host.into(),
            device: device.into(),
        }
    }

    /// Replicas take equal consecutive blocks of the stated ranks and are
    /// named by their index; components without ranks span each replica.
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

            [worker_groups.components.transformer]
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
            let transformer = &worker.components["transformer"];
            assert_eq!(transformer.ranks, vec![0, 1]);
            assert_eq!(transformer.parallel_config.world_size().unwrap(), 2);
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

            [worker_groups.components.muxer]
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

    /// The video decoder and codec are temporal-unit components whether or
    /// not the file says so, and no other unit count is a placement they
    /// have.
    #[test]
    fn the_video_decoder_and_codec_take_one_unit_per_rank() {
        let decoder = |table: &str| {
            workers(&format!(
                "[[worker_groups]]\nid = \"model\"\nranks = [\"cuda:0-1\"]\nqueue_depth = 2\n\
                 [worker_groups.components.video_decoder]\n{table}"
            ))
        };

        let omitted = decoder("").unwrap();
        let stated = decoder("distribution = \"temporal_units\"").unwrap();
        assert_eq!(omitted, stated);
        assert_eq!(
            omitted[0].components["video_decoder"].distribution,
            Some(ComponentDistribution::TemporalUnits)
        );
        assert!(decoder("units_per_rank = 2").is_err());
    }

    #[test]
    fn malformed_deployments_are_refused() {
        let group =
            |body: &str| format!("[[worker_groups]]\nid = \"model\"\nqueue_depth = 2\n{body}");
        for refused in [
            // Unknown keys, at the group and the component.
            group("ranks = [\"cuda:0\"]\nreplica = 2"),
            group("ranks = [\"cuda:0\"]\n[worker_groups.components.model]\ntensor_parallel = 1"),
            // Rank entries that name no devices.
            group("ranks = [\"cuda:3-1\"]"),
            group("ranks = [\"cpu*0\"]"),
            group("ranks = [\"/cuda:0\"]"),
            // Replicas that do not divide the ranks.
            group("ranks = [\"cuda:0-2\"]\nreplicas = 2"),
            // A component whose degrees do not match its ranks.
            group(
                "ranks = [\"cuda:0-1\"]\n[worker_groups.components.model]\ntensor_parallel_size = 4",
            ),
            // No groups at all.
            String::new(),
        ] {
            assert!(workers(&refused).is_err(), "{refused:?} must be refused");
        }
    }

    /// Every deployment the repository publishes for `--worker-groups` places.
    #[test]
    fn published_deployments_place() {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../configs");
        let mut files = 0;
        for model in std::fs::read_dir(root).unwrap() {
            for entry in std::fs::read_dir(model.unwrap().path()).unwrap() {
                let path = entry.unwrap().path();
                let deployment = Deployment::read(path.to_str().unwrap()).unwrap();
                deployment
                    .worker_groups("localhost")
                    .unwrap_or_else(|error| panic!("{}: {error:#}", path.display()));
                files += 1;
            }
        }
        assert!(files > 0, "configs/ publishes deployments");
    }
}

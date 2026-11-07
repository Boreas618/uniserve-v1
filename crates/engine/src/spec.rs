//! The engine configuration an [`EngineSpec`] states.
//!
//! The leader daemon resolves the serving options and the model's facts into
//! the spec it hands a main process; this module places the deployment's
//! ranks (from its deployment file or the rank layout shorthand), and reads the
//! words the options name values by (element types, the expert exchange) into
//! the engine's own values.

use anyhow::Context as _;
use proto::request::GenerationLimits;
use proto::request::wire::{EngineSpec, RankLayout, Sizes};
use serde::de::DeserializeOwned;

use crate::DEFAULT_COMPONENT;
use crate::core::{EngineConfig, ExecutorGroupConfig};
use crate::deployment::Deployment;
use crate::executor::TransferConfig;
use crate::scheduler::{DEFAULT_CHUNKED_PREFILL_SIZE, DEFAULT_MAX_NUM_BATCHED_TOKENS};

/// The value the word `value` names for the option at `option`, read as the
/// serving options name it.
fn named<T: DeserializeOwned>(value: &str, option: &str) -> anyhow::Result<T> {
    serde_json::from_value(serde_json::Value::String(value.to_owned()))
        .with_context(|| format!("invalid `{option}`: {value:?}"))
}

/// A list the spec states, `None` when absent.
fn sizes(list: &Option<Sizes>) -> Option<Vec<u32>> {
    list.as_ref().map(|list| list.values.clone())
}

impl EngineConfig {
    /// Resolves independent data-parallel replicas into standalone engine
    /// specs with explicit host and GPU assignments. Expert-parallel groups
    /// communicate across schedulers and remain in one engine spec.
    ///
    /// Fails for invalid engine options, unequal scheduler groups, or a
    /// transfer edge between otherwise independent replicas.
    pub fn replica_specs(spec: &EngineSpec) -> anyhow::Result<Vec<EngineSpec>> {
        let config = Self::from_spec(spec)?;
        let count = config.data_parallel_size;
        if count == 1 || config.expert_parallel.is_some() {
            return Ok(vec![spec.clone()]);
        }
        anyhow::ensure!(
            !config.worker_groups.is_empty() && config.worker_groups.len().is_multiple_of(count),
            "executor groups must form {count} equal independent replicas"
        );
        let groups_per_replica = config.worker_groups.len() / count;
        let transfers = TransferConfig::parse(&spec.transfer)?;
        config
            .worker_groups
            .chunks(groups_per_replica)
            .map(|groups| {
                let contains = |id: &crate::executor::ExecutorGroupId| {
                    groups.iter().any(|group| &group.id == id)
                };
                let mut local = TransferConfig::default();
                for edge in &transfers.edges {
                    let source = contains(&edge.source_worker);
                    let destination = contains(&edge.destination_worker);
                    anyhow::ensure!(
                        source == destination,
                        "a transfer edge connects independent replicas: {} -> {}",
                        edge.source_worker,
                        edge.destination_worker
                    );
                    if source {
                        local.edges.push(edge.clone());
                    }
                }
                let mut replica = spec.clone();
                let ranks = replica
                    .rank_layout
                    .as_mut()
                    .context("the engine spec states no ranks")?;
                ranks.data_parallel_size = 1;
                ranks.worker_groups.clear();
                ranks.worker_groups_content = Deployment::snapshot(groups)?;
                replica.transfer = local.to_string();
                Ok(replica)
            })
            .collect()
    }

    /// The configuration `spec` states.
    ///
    /// # Errors
    ///
    /// Fails, naming the option, when the spec omits a table, names a value
    /// the engine does not know, its deployment file does not read or place,
    /// the transfer edges do not resolve, or the precision policy is not a
    /// JSON object.
    pub fn from_spec(spec: &EngineSpec) -> anyhow::Result<Self> {
        let rank_layout = required(&spec.rank_layout, "rank layout")?;
        let scheduling = required(&spec.scheduling, "scheduling")?;
        let kv_cache = required(&spec.kv_cache, "kv_cache")?;
        let workers = required(&spec.workers, "executors")?;
        let graphs = required(&spec.cuda_graphs, "cuda_graphs")?;
        let experts = required(&spec.experts, "experts")?;
        let facts = required(&spec.model_facts, "model_facts")?;

        let worker_groups = place(rank_layout, &spec.host)?;
        spec.transfer
            .parse::<TransferConfig>()
            .context("invalid `device_config.transfer`")?;
        let max_num_batched_tokens = scheduling
            .max_num_batched_tokens
            .map_or(DEFAULT_MAX_NUM_BATCHED_TOKENS, |tokens| tokens as usize);
        let host_kv_cache = kv_cache.prefix_caching && kv_cache.host_tier;

        Ok(Self {
            generation_limits: facts.limits.map(GenerationLimits::from).unwrap_or_default(),
            max_batch: scheduling.max_batch.max(1) as usize,
            max_num_batched_tokens,
            max_running_requests: scheduling.max_running_requests.max(1) as usize,
            max_queued_requests: scheduling
                .max_queued_requests
                .map(|requests| requests as usize),
            chunked_prefill_size: scheduling
                .chunked_prefill_size
                .map_or(DEFAULT_CHUNKED_PREFILL_SIZE, |size| size as usize),
            graph_tiers: {
                // The deployment's captured graph catalog, ascending and
                // without duplicates: the largest of them bounds a batch's
                // tokens (`Room.graph_tiers`).
                let mut tiers: Vec<u32> = sizes(&graphs.decode_batch_sizes).unwrap_or_default();
                tiers.extend(sizes(&graphs.prefill_token_sizes).unwrap_or_default());
                tiers.sort_unstable();
                tiers.dedup();
                tiers
            },
            prefix_cache: kv_cache.prefix_caching,
            boundary_interval: kv_cache
                .boundary_interval
                .filter(|interval| *interval > 0)
                .map_or(crate::kv_cache::DEFAULT_BOUNDARY_INTERVAL, |interval| {
                    interval as usize
                }),
            state_interval: kv_cache
                .state_interval
                .filter(|interval| *interval > 0)
                .map_or(crate::kv_cache::DEFAULT_STATE_INTERVAL, |interval| {
                    interval as usize
                }),
            host_cache: crate::kv_cache::HostCacheSize::new(
                kv_cache.host_ratio,
                kv_cache.host_bytes,
            )?,
            worker_groups,
            data_parallel_size: rank_layout.data_parallel_size.max(1) as usize,
            expert_parallel: experts
                .expert_parallel
                .then(|| named(&experts.exchange, "moe_config.exchange"))
                .transpose()?,
            model: spec.model.clone(),
            dtype: named(&workers.dtype, "dtype")?,
            // A deployment is what one daemon serves under its served name.
            deployment: spec.deployment.clone(),
            host: spec.host.clone(),
            host_kv_cache,
            kv_storage: host_kv_cache && kv_cache.mooncake.is_some(),
            canvas_sampling: facts
                .canvas_sampling
                .as_ref()
                .map(|canvas| ipc::CanvasSampling {
                    canvas_length: canvas.canvas_length,
                    max_steps: canvas.max_steps,
                    entropy_bound: canvas.entropy_bound,
                    t_min: canvas.t_min,
                    t_max: canvas.t_max,
                    confidence_threshold: canvas.confidence_threshold,
                    stability_threshold: canvas.stability_threshold,
                }),
            draft_tokens: spec
                .speculation
                .as_ref()
                .map_or(0, |speculation| speculation.num_draft_tokens),
            bos: facts.bos,
            eos: facts.eos.clone(),
            end_of_image: facts.end_of_image,
            world: spec.world,
            trial: spec.predecessor.is_some(),
        })
    }
}

/// The table `name` of a spec, which every spec states.
fn required<'a, T>(table: &'a Option<T>, name: &str) -> anyhow::Result<&'a T> {
    table
        .as_ref()
        .with_context(|| format!("the engine spec states no `{name}`"))
}

/// The deployment's executor groups: its deployment file placed on `host`, or
/// the shorthand's groups of one component, with the ranks spread in blocks
/// over `host` and the hosts after it.
fn place(rank_layout: &RankLayout, host: &str) -> anyhow::Result<Vec<ExecutorGroupConfig>> {
    if !rank_layout.worker_groups_content.is_empty() {
        return Deployment::parse(&rank_layout.worker_groups_content)?
            .worker_groups(host)
            .context("invalid `worker_groups` deployment");
    }
    // Leading with the leader's own host keeps rank zero and the other
    // lowest ranks there; a repeat of it, or an empty name, places nothing.
    let mut hosts = vec![host.to_owned()];
    hosts.extend(
        rank_layout
            .worker_hosts
            .iter()
            .filter(|worker_host| !worker_host.is_empty() && *worker_host != host)
            .cloned(),
    );
    let tensor_parallel_size = rank_layout.tensor_parallel_size.max(1) as usize;
    let mut groups = ExecutorGroupConfig::data_parallel(
        &hosts,
        &rank_layout.device,
        rank_layout.data_parallel_size.max(1) as usize,
        tensor_parallel_size,
        rank_layout.queue_depth.max(1) as usize,
        ExecutorGroupConfig::single_component(DEFAULT_COMPONENT, tensor_parallel_size),
    );
    if !rank_layout.devices.is_empty() {
        order_devices(&mut groups, &rank_layout.device, &rank_layout.devices)?;
    }
    Ok(groups)
}

/// Puts the shorthand's ranks, in rank order across `groups`, on the CUDA
/// devices `devices` names by their index on each rank's host
/// (`device_config.devices`): a world whose ranks a daemon assigned to
/// the GPUs that already hold most of their weights, or an operator's
/// order.
///
/// # Errors
///
/// Fails unless `device` is CUDA, `devices` names one device per rank, and
/// no two ranks of a host share a device.
fn order_devices(
    groups: &mut [ExecutorGroupConfig],
    device: &str,
    devices: &[u32],
) -> anyhow::Result<()> {
    anyhow::ensure!(
        matches!(device, "cuda" | "gpu"),
        "`device_config.devices` orders CUDA devices; `device_config.device` is {device:?}"
    );
    let ranks: usize = groups.iter().map(|group| group.ranks.len()).sum();
    anyhow::ensure!(
        devices.len() == ranks,
        "`device_config.devices` names {} devices for {ranks} ranks",
        devices.len()
    );
    let mut taken = std::collections::BTreeSet::new();
    let ranks = groups.iter_mut().flat_map(|group| group.ranks.iter_mut());
    for (rank, index) in ranks.zip(devices) {
        anyhow::ensure!(
            taken.insert((rank.host.clone(), *index)),
            "`device_config.devices` puts two ranks of host {} on device {index}",
            rank.host
        );
        rank.device = format!("cuda:{index}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use proto::request::wire;

    use super::*;

    /// A spec of one rank on this host under every option's default word.
    fn spec() -> EngineSpec {
        EngineSpec {
            deployment: "model".into(),
            model: "/models/model".into(),
            host: "localhost".into(),
            rank_layout: Some(RankLayout {
                tensor_parallel_size: 1,
                data_parallel_size: 1,
                device: "cuda".into(),
                queue_depth: 2,
                mem_fraction_static: 0.7,
                ..RankLayout::default()
            }),
            transfer: String::new(),
            scheduling: Some(wire::Scheduling {
                max_batch: 128,
                max_running_requests: 128,
                ..wire::Scheduling::default()
            }),
            kv_cache: Some(wire::KvCache::default()),
            workers: Some(wire::Workers {
                executable: "uniserve".into(),
                watchdog_timeout_ms: 300_000,
                attention_backend: "auto".into(),
                dtype: "bfloat16".into(),
                load_format: "auto".into(),
                ..wire::Workers::default()
            }),
            cuda_graphs: Some(wire::CudaGraphs {
                policy: "auto".into(),
                ..wire::CudaGraphs::default()
            }),
            video: Some(wire::Video {
                max_seconds: 15.0,
                ffmpeg: "ffmpeg".into(),
                ..wire::Video::default()
            }),
            experts: Some(wire::Experts {
                exchange: "alltoall".into(),
                microbatches: 1,
                ..wire::Experts::default()
            }),
            model_facts: Some(wire::ModelFacts::default()),
            world: 0,
            predecessor: None,
            stage_cache: false,
            cache_keys: Vec::new(),
            segments: Vec::new(),
            retire: Vec::new(),
            speculation: None,
            profiling: None,
            received_weights: Vec::new(),
            expert_order: Default::default(),
        }
    }

    /// A word no engine value takes is refused, naming its option.
    #[test]
    fn an_unknown_word_is_refused_naming_its_option() {
        let mut unknown = spec();
        unknown.workers.as_mut().unwrap().dtype = "sometimes".into();
        let error = EngineConfig::from_spec(&unknown).expect_err("the word must be refused");
        assert!(format!("{error:#}").contains("dtype"), "{error:#}");
    }

    /// A spec of `ranks` tensor-parallel ranks sharing their experts, on
    /// the devices `devices` orders.
    fn tensor_experts(ranks: u32, devices: &[u32]) -> EngineSpec {
        let mut spec = spec();
        let shorthand = spec.rank_layout.as_mut().unwrap();
        shorthand.tensor_parallel_size = ranks;
        shorthand.devices = devices.to_vec();
        spec.experts.as_mut().unwrap().expert_parallel = true;
        spec
    }

    /// The devices of `config`'s groups, group by group in rank order.
    fn devices(config: &EngineConfig) -> Vec<(String, Vec<String>)> {
        config
            .worker_groups
            .iter()
            .map(|group| {
                (
                    group.id.to_string(),
                    group.ranks.iter().map(|rank| rank.device.clone()).collect(),
                )
            })
            .collect()
    }

    /// The shorthand puts its ranks, in rank order, on the devices
    /// `device_config.devices` names.
    #[test]
    fn the_shorthand_puts_ranks_on_the_devices_their_order_names() {
        let config = EngineConfig::from_spec(&tensor_experts(4, &[0, 2, 1, 3])).unwrap();
        assert_eq!(
            devices(&config),
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
        let config = EngineConfig::from_spec(&tensor_experts(4, &[])).unwrap();
        assert_eq!(
            devices(&config),
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

    /// Schedulers that share their experts keep the order across their groups:
    /// the devices name each scheduler's rank in turn.
    #[test]
    fn a_device_order_runs_across_the_schedulers_groups() {
        let mut spec = spec();
        let shorthand = spec.rank_layout.as_mut().unwrap();
        shorthand.data_parallel_size = 2;
        shorthand.devices = vec![1, 0];
        spec.experts.as_mut().unwrap().expert_parallel = true;
        let config = EngineConfig::from_spec(&spec).unwrap();
        assert_eq!(
            devices(&config),
            [
                ("model-0".to_owned(), vec!["cuda:1".to_owned()]),
                ("model-1".to_owned(), vec!["cuda:0".to_owned()]),
            ]
        );
    }

    /// An order that does not name one CUDA device per rank, or puts two
    /// ranks of a host on one device, is refused naming the option.
    #[test]
    fn a_device_order_of_another_length_or_a_repeated_device_is_refused() {
        for devices in [&[0, 1, 2][..], &[0, 1, 1, 2]] {
            let error = EngineConfig::from_spec(&tensor_experts(4, devices))
                .expect_err("the order must be refused");
            assert!(
                format!("{error:#}").contains("device_config.devices"),
                "{error:#}"
            );
        }
        let mut cpu = tensor_experts(2, &[0, 1]);
        cpu.rank_layout.as_mut().unwrap().device = "cpu".into();
        assert!(EngineConfig::from_spec(&cpu).is_err());
    }

    #[test]
    fn independent_replicas_preserve_their_assigned_tensor_parallel_ranks() {
        let mut deployment = spec();
        let ranks = deployment.rank_layout.as_mut().unwrap();
        ranks.tensor_parallel_size = 2;
        ranks.data_parallel_size = 2;
        let replicas = EngineConfig::replica_specs(&deployment).unwrap();
        let devices: Vec<Vec<String>> = replicas
            .iter()
            .map(|replica| {
                EngineConfig::from_spec(replica)
                    .unwrap()
                    .worker_groups
                    .into_iter()
                    .flat_map(|group| group.ranks.into_iter().map(|rank| rank.device))
                    .collect()
            })
            .collect();
        assert_eq!(
            devices,
            vec![vec!["cuda:0", "cuda:1"], vec!["cuda:2", "cuda:3"]]
        );
        assert!(
            replicas
                .iter()
                .all(|replica| replica.scheduling == deployment.scheduling)
        );
    }

    #[test]
    fn communicating_expert_ranks_stay_in_one_replica() {
        let mut deployment = spec();
        deployment.rank_layout.as_mut().unwrap().data_parallel_size = 4;
        deployment.experts.as_mut().unwrap().expert_parallel = true;
        let replicas = EngineConfig::replica_specs(&deployment).unwrap();
        assert_eq!(replicas.len(), 1);
        assert_eq!(
            EngineConfig::from_spec(&replicas[0])
                .unwrap()
                .worker_groups
                .iter()
                .map(|group| group.ranks.len())
                .sum::<usize>(),
            4
        );
    }
}

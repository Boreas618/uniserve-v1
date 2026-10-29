//! The engine configuration an [`EngineSpec`] states.
//!
//! The leader daemon resolves the serving options and the model's facts into
//! the spec it hands the engine process; this module places the deployment's
//! ranks (from its deployment file or the placement shorthand), reads the
//! words the options name values by (policies, backends, element types) into
//! the engine's own values, and completes every worker's launch arguments.

use std::num::NonZeroU64;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Context as _;
use proto::request::GenerationLimits;
use proto::request::wire::{CudaGraphs, EngineSpec, FlashInfer, Placement, Scheduling, Sizes};
use serde::de::DeserializeOwned;

use crate::DEFAULT_COMPONENT;
use crate::core::{EngineConfig, WorkerGroupConfig};
use crate::deployment::Deployment;
use crate::executor::TransferConfig;
use crate::scheduler::{
    DEFAULT_CHUNKED_PREFILL_SIZE, DEFAULT_MAX_NUM_BATCHED_TOKENS, RefillPolicy, SchedulerConfig,
};
use crate::worker::WorkerProcessArgs;

/// The value the word `value` names for the option at `option`, read as the
/// serving options name it.
fn named<T: DeserializeOwned>(value: &str, option: &str) -> anyhow::Result<T> {
    serde_json::from_value(serde_json::Value::String(value.to_owned()))
        .with_context(|| format!("invalid `{option}`: {value:?}"))
}

/// The scheduling limits and policy `scheduling` states, with the engine's
/// defaults for those it leaves out; the canvas is the scheduler's own.
///
/// # Errors
///
/// Fails on a policy the engine does not know.
pub(crate) fn scheduler_limits(scheduling: &Scheduling) -> anyhow::Result<SchedulerConfig> {
    Ok(SchedulerConfig {
        policy: named(&scheduling.policy, "scheduler_config.policy")?,
        max_batch: scheduling.max_batch.max(1) as usize,
        max_num_batched_tokens: scheduling
            .max_num_batched_tokens
            .map_or(DEFAULT_MAX_NUM_BATCHED_TOKENS, |tokens| tokens as usize),
        max_running_requests: scheduling.max_running_requests.max(1) as usize,
        chunked_prefill_size: scheduling
            .chunked_prefill_size
            .map_or(DEFAULT_CHUNKED_PREFILL_SIZE, |size| size as usize),
        max_queued_requests: scheduling
            .max_queued_requests
            .map(|requests| requests as usize),
        canvas: None,
        draft_tokens: 0,
        refill: refill_policy(scheduling),
    })
}

/// The draft cache refill policy `scheduling` states: the values it states
/// in place of the ones the world measures.
pub(crate) fn refill_policy(scheduling: &Scheduling) -> RefillPolicy {
    RefillPolicy {
        expected_speedup: scheduling.expected_speedup,
        token_cost: scheduling.refill_token_cost,
    }
}

/// A list the spec states, `None` when absent.
fn sizes(list: &Option<Sizes>) -> Option<Vec<u32>> {
    list.as_ref().map(|list| list.values.clone())
}

/// An optional text the spec states as empty when unset.
fn stated(text: &str) -> Option<String> {
    (!text.is_empty()).then(|| text.to_owned())
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
            "worker groups must form {count} equal independent replicas"
        );
        let groups_per_replica = config.worker_groups.len() / count;
        let transfers = TransferConfig::parse(&spec.transfer)?;
        config
            .worker_groups
            .chunks(groups_per_replica)
            .map(|groups| {
                let contains = |id: &crate::executor::WorkerGroupId| {
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
                    .placement
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
        let placement = required(&spec.placement, "placement")?;
        let scheduling = required(&spec.scheduling, "scheduling")?;
        let kv_cache = required(&spec.kv_cache, "kv_cache")?;
        let workers = required(&spec.workers, "workers")?;
        let graphs = required(&spec.cuda_graphs, "cuda_graphs")?;
        let flashinfer = required(&spec.flashinfer, "flashinfer")?;
        let video = required(&spec.video, "video")?;
        let experts = required(&spec.experts, "experts")?;
        let facts = required(&spec.model_facts, "model_facts")?;

        let worker_groups = place(placement, &spec.host)?;
        let transfer = if spec.transfer.is_empty() {
            TransferConfig::default()
        } else {
            spec.transfer
                .parse::<TransferConfig>()
                .context("invalid `placement_config.transfer`")?
        };
        let quantization_config: serde_json::Value = if workers.quantization_config.is_empty() {
            serde_json::Value::Object(serde_json::Map::new())
        } else {
            serde_json::from_str(&workers.quantization_config)
                .context("invalid `quantization_config`")?
        };
        anyhow::ensure!(
            quantization_config.is_object(),
            "`quantization_config` must be a table"
        );

        let max_num_batched_tokens = scheduling
            .max_num_batched_tokens
            .map_or(DEFAULT_MAX_NUM_BATCHED_TOKENS, |tokens| tokens as usize);
        let host_kv_cache = kv_cache.prefix_caching && kv_cache.host_tier;
        let cache_keys = spec
            .cache_keys
            .iter()
            .map(|key| {
                <[u8; 32]>::try_from(key.as_slice())
                    .context("a retained cache key must contain 32 bytes")
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let worker_process = WorkerProcessArgs {
            executable: PathBuf::from(&workers.executable),
            // A deployment is what one daemon serves under its served name.
            deployment: spec.deployment.clone(),
            host_kv_cache,
            kv_storage: kv_cache
                .mooncake
                .as_ref()
                .filter(|_| host_kv_cache)
                .map(|mooncake| storage::connector::Settings {
                    master: mooncake.master.clone(),
                    metadata: mooncake.metadata_server.clone(),
                    protocol: mooncake.protocol.clone(),
                    device: mooncake.device.clone(),
                    // Each host names itself to Mooncake.
                    hostname: None,
                    segment_bytes: mooncake.segment_bytes,
                    library: stated(&mooncake.library).map(PathBuf::from),
                }),
            keep_standby: workers.standby,
            standby: false,
            standby_storage: if workers.standby_storage.is_empty() {
                proto::request::StandbyStorage::Auto
            } else {
                named(&workers.standby_storage, "worker_config.standby_storage")?
            },
            segments: None,
            world: spec.world,
            retire: spec.retire.clone(),
            predecessor: None,
            stage_cache: spec.stage_cache,
            cache_keys: cache_keys.clone(),
            received_weights: None,
            model: spec.model.clone(),
            expert_order: spec
                .expert_order
                .iter()
                .map(|(path, order)| (path.clone(), order.experts.clone()))
                .collect(),
            weights: stated(&workers.weights),
            host: spec.host.clone(),
            queue_depth: placement.queue_depth.max(1) as usize,
            watchdog_timeout: Duration::from_millis(workers.watchdog_timeout_ms),
            max_total_tokens: kv_cache
                .max_tokens
                .and_then(NonZeroU64::new)
                .map(NonZeroU64::get),
            block_size: kv_cache.block_size,
            attention_backend: workers.attention_backend.clone(),
            expert_microbatches: experts.microbatches.max(1),
            load_format: named(&workers.load_format, "load_config.format")?,
            download_dir: stated(&workers.download_dir).map(PathBuf::from),
            load_threads: workers.load_threads,
            parallel_degrees: workers.parallel_degrees.clone(),
            dtype: named(&workers.dtype, "dtype")?,
            quantization_config,
            kv_cache_dtype: stated(&kv_cache.dtype)
                .map(|dtype| named(&dtype, "kv_cache_config.dtype"))
                .transpose()?,
            mem_fraction_static: placement.mem_fraction_static,
            worker_mesh: stated(&workers.mesh),
            distributed_backend: stated(&workers.distributed_backend),
            max_model_len: facts.max_model_len,
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
            prefill_outputs: facts.prefill_outputs,
            draft_tokens: spec
                .speculation
                .as_ref()
                .map_or(0, |speculation| speculation.num_draft_tokens),
            video_text_capacities: sizes(&video.text_capacities),
            ffmpeg: PathBuf::from(&video.ffmpeg),
            // Each forward batch is bounded by the scheduler's batch and
            // token budgets, which the worker's launch descriptor carries.
            max_batch_calls: scheduling.max_batch,
            max_batch_tokens: u32::try_from(max_num_batched_tokens)
                .context("`max_num_batched_tokens` exceeds the worker field width")?,
            max_video_seconds: video.max_seconds,
            min_video_seconds: (video.min_seconds > 0.0).then_some(video.min_seconds),
            max_condition_rows: video.max_condition_rows,
            video_resolutions: video.resolutions.clone(),
            video_aspect_ratios: video.aspect_ratios.clone(),
            fence_probes: spec
                .profiling
                .as_ref()
                .is_some_and(|profiling| profiling.fence_probes),
            fence_components: spec
                .profiling
                .as_ref()
                .map(|profiling| profiling.fence_components.clone())
                .unwrap_or_default(),
            fence_probes_off: spec
                .profiling
                .as_ref()
                .map(|profiling| profiling.fence_probes_off.clone())
                .unwrap_or_default(),
            graph_pc_sampling: spec
                .profiling
                .as_ref()
                .is_some_and(|profiling| profiling.graph_pc_sampling),
            ..graph_arguments(graphs, flashinfer)?
        };

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
            refill: refill_policy(scheduling),
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
            schedule_policy: named(&scheduling.policy, "scheduler_config.policy")?,
            worker_groups,
            transfer,
            data_parallel_size: placement.data_parallel_size.max(1) as usize,
            expert_parallel: experts
                .expert_parallel
                .then(|| named(&experts.exchange, "moe_config.exchange"))
                .transpose()?,
            worker_process,
            bos: facts.bos,
            eos: facts.eos.clone(),
            end_of_image: facts.end_of_image,
            world: spec.world,
            stage_cache: spec.stage_cache,
            cache_keys,
            predecessor: spec.predecessor.as_ref().map(|predecessor| {
                predecessor
                    .segments
                    .iter()
                    .map(|group| {
                        (
                            group.worker_group.clone(),
                            ipc::PredecessorCache {
                                world: predecessor.world,
                                segments: group
                                    .segments
                                    .iter()
                                    .take(group.held as usize)
                                    .map(|segment| ipc::CacheSegment {
                                        world: segment.world,
                                        ordinal: segment.ordinal,
                                        units: segment.units,
                                    })
                                    .collect(),
                                num_units: group.units,
                                scratch_units: group.scratch,
                            },
                        )
                    })
                    .collect()
            }),
            segments: spec
                .segments
                .iter()
                .map(|group| {
                    (
                        group.worker_group.clone(),
                        ipc::CacheTable {
                            segments: group
                                .segments
                                .iter()
                                .map(|segment| ipc::CacheSegment {
                                    world: segment.world,
                                    ordinal: segment.ordinal,
                                    units: segment.units,
                                })
                                .collect(),
                            held: group.held,
                        },
                    )
                })
                .collect(),
            received_weights: spec
                .received_weights
                .iter()
                .map(|group| {
                    Ok((
                        group.worker_group.clone(),
                        ipc::ReceivedWeights {
                            worlds: group.worlds.clone(),
                            manifest: group.manifest.clone(),
                        },
                    ))
                })
                .collect::<anyhow::Result<_>>()?,
        })
    }
}

/// The table `name` of a spec, which every spec states.
fn required<'a, T>(table: &'a Option<T>, name: &str) -> anyhow::Result<&'a T> {
    table
        .as_ref()
        .with_context(|| format!("the engine spec states no `{name}`"))
}

/// The deployment's worker groups: its deployment file placed on `host`, or
/// the shorthand's groups of one component, with the ranks spread in blocks
/// over `host` and the hosts after it.
fn place(placement: &Placement, host: &str) -> anyhow::Result<Vec<WorkerGroupConfig>> {
    if !placement.worker_groups_content.is_empty() {
        return Deployment::parse(&placement.worker_groups_content)?
            .worker_groups(host)
            .context("invalid `worker_groups` deployment");
    }
    // Leading with the leader's own host keeps rank zero and the other
    // lowest ranks there; a repeat of it, or an empty name, places nothing.
    let mut hosts = vec![host.to_owned()];
    hosts.extend(
        placement
            .worker_hosts
            .iter()
            .filter(|worker_host| !worker_host.is_empty() && *worker_host != host)
            .cloned(),
    );
    let tensor_parallel_size = placement.tensor_parallel_size.max(1) as usize;
    let mut groups = WorkerGroupConfig::data_parallel(
        &hosts,
        &placement.device,
        placement.data_parallel_size.max(1) as usize,
        tensor_parallel_size,
        placement.queue_depth.max(1) as usize,
        WorkerGroupConfig::single_component(DEFAULT_COMPONENT, tensor_parallel_size),
    );
    if !placement.devices.is_empty() {
        order_devices(&mut groups, &placement.device, &placement.devices)?;
    }
    Ok(groups)
}

/// Puts the shorthand's ranks, in rank order across `groups`, on the CUDA
/// devices `devices` names by their index on each rank's host
/// (`placement_config.devices`): a world whose ranks a daemon assigned to
/// the GPUs that already hold most of their weights, or an operator's
/// order.
///
/// # Errors
///
/// Fails unless `device` is CUDA, `devices` names one device per rank, and
/// no two ranks of a host share a device.
fn order_devices(
    groups: &mut [WorkerGroupConfig],
    device: &str,
    devices: &[u32],
) -> anyhow::Result<()> {
    anyhow::ensure!(
        matches!(device, "cuda" | "gpu"),
        "`placement_config.devices` orders CUDA devices; `placement_config.device` is {device:?}"
    );
    let ranks: usize = groups.iter().map(|group| group.ranks.len()).sum();
    anyhow::ensure!(
        devices.len() == ranks,
        "`placement_config.devices` names {} devices for {ranks} ranks",
        devices.len()
    );
    let mut taken = std::collections::BTreeSet::new();
    let ranks = groups.iter_mut().flat_map(|group| group.ranks.iter_mut());
    for (rank, index) in ranks.zip(devices) {
        anyhow::ensure!(
            taken.insert((rank.host.clone(), *index)),
            "`placement_config.devices` puts two ranks of host {} on device {index}",
            rank.host
        );
        rank.device = format!("cuda:{index}");
    }
    Ok(())
}

/// The launch arguments the graph and attention-provider tables state.
fn graph_arguments(
    graphs: &CudaGraphs,
    flashinfer: &FlashInfer,
) -> anyhow::Result<WorkerProcessArgs> {
    Ok(WorkerProcessArgs {
        graph_policy: named(&graphs.policy, "cuda_graph_config.policy")?,
        decode_graph_batch_sizes: sizes(&graphs.decode_batch_sizes),
        prefill_graph_token_sizes: sizes(&graphs.prefill_token_sizes),
        flow_graph_shapes: graphs.flow_shapes.as_ref().map(|shapes| {
            shapes
                .shapes
                .iter()
                .map(|shape| [shape.height, shape.width])
                .collect()
        }),
        flashinfer_workspace_size: flashinfer.workspace_size,
        flashinfer_use_tensor_core: flashinfer.use_tensor_core,
        flashinfer_decode_backend: named(
            &flashinfer.decode_backend,
            "flashinfer_config.decode_backend",
        )?,
        flashinfer_prefill_backend: named(
            &flashinfer.prefill_backend,
            "flashinfer_config.prefill_backend",
        )?,
        flashinfer_decode_split_tile_size: flashinfer.decode_split_tile_size,
        flashinfer_prefill_split_tile_size: flashinfer.prefill_split_tile_size,
        flashinfer_split_kv: flashinfer.split_kv,
        ..WorkerProcessArgs::default()
    })
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
            placement: Some(Placement {
                tensor_parallel_size: 1,
                data_parallel_size: 1,
                device: "cuda".into(),
                queue_depth: 2,
                mem_fraction_static: 0.7,
                ..Placement::default()
            }),
            transfer: String::new(),
            scheduling: Some(wire::Scheduling {
                policy: "fcfs".into(),
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
            cuda_graphs: Some(CudaGraphs {
                policy: "auto".into(),
                ..CudaGraphs::default()
            }),
            flashinfer: Some(FlashInfer {
                decode_backend: "fa2".into(),
                prefill_backend: "auto".into(),
                ..FlashInfer::default()
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
        unknown.cuda_graphs.as_mut().unwrap().policy = "sometimes".into();
        let error = EngineConfig::from_spec(&unknown).expect_err("the word must be refused");
        assert!(
            format!("{error:#}").contains("cuda_graph_config.policy"),
            "{error:#}"
        );
    }

    /// A spec of `ranks` tensor-parallel ranks sharing their experts, on
    /// the devices `devices` orders.
    fn tensor_experts(ranks: u32, devices: &[u32]) -> EngineSpec {
        let mut spec = spec();
        let shorthand = spec.placement.as_mut().unwrap();
        shorthand.tensor_parallel_size = ranks;
        shorthand.devices = devices.to_vec();
        spec.experts.as_mut().unwrap().expert_parallel = true;
        spec
    }

    /// The shorthand puts its ranks, in rank order, on the devices
    /// `placement_config.devices` names; a planned world's launches state
    /// them, with the ranks each loads its weights with.
    #[test]
    fn the_shorthand_puts_ranks_on_the_devices_their_order_names() {
        let launches = crate::launches::rank_launches(&tensor_experts(4, &[0, 2, 1, 3])).unwrap();
        let devices: Vec<&str> = launches
            .iter()
            .map(|launch| launch.descriptor.device.as_str())
            .collect();
        assert_eq!(devices, ["cuda:0", "cuda:2", "cuda:1", "cuda:3"]);
        for (index, launch) in launches.iter().enumerate() {
            assert_eq!(launch.rank, index as u32);
            assert_eq!(launch.descriptor.rank, index as u32);
            assert_eq!(
                launch.agreement,
                crate::launches::Agreement::Group("model".into())
            );
            assert_eq!(launch.agreement_index, index as u32);
            assert!(launch.descriptor.tensor_experts);
            assert_eq!(launch.environment["LOCAL_RANK"], index.to_string());
        }

        // Without an order each host's ranks number its devices from zero.
        let launches = crate::launches::rank_launches(&tensor_experts(4, &[])).unwrap();
        let devices: Vec<&str> = launches
            .iter()
            .map(|launch| launch.descriptor.device.as_str())
            .collect();
        assert_eq!(devices, ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]);
    }

    /// Schedulers that share their experts load their weights together
    /// over the expert union, each at its union rank.
    #[test]
    fn the_ranks_of_an_expert_union_load_together() {
        let mut spec = spec();
        let shorthand = spec.placement.as_mut().unwrap();
        shorthand.data_parallel_size = 2;
        shorthand.devices = vec![1, 0];
        spec.experts.as_mut().unwrap().expert_parallel = true;
        let launches = crate::launches::rank_launches(&spec).unwrap();
        let planned: Vec<(&str, &str, u32)> = launches
            .iter()
            .map(|launch| {
                assert_eq!(launch.agreement, crate::launches::Agreement::ExpertUnion);
                (
                    launch.group.as_str(),
                    launch.descriptor.device.as_str(),
                    launch.agreement_index,
                )
            })
            .collect();
        assert_eq!(
            planned,
            [("model-0", "cuda:1", 0), ("model-1", "cuda:0", 1)]
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
                format!("{error:#}").contains("placement_config.devices"),
                "{error:#}"
            );
        }
        let mut cpu = tensor_experts(2, &[0, 1]);
        cpu.placement.as_mut().unwrap().device = "cpu".into();
        assert!(EngineConfig::from_spec(&cpu).is_err());
    }

    #[test]
    fn independent_replicas_preserve_their_assigned_tensor_parallel_ranks() {
        let mut deployment = spec();
        let ranks = deployment.placement.as_mut().unwrap();
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
        deployment.placement.as_mut().unwrap().data_parallel_size = 4;
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

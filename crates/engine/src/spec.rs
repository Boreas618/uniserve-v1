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
            standby_memory: if workers.standby_memory.is_empty() {
                proto::request::StandbyMemory::Auto
            } else {
                named(&workers.standby_memory, "worker_config.standby_memory")?
            },
            segments: None,
            world: spec.world,
            retire: spec.retire.clone(),
            predecessor: None,
            model: spec.model.clone(),
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
    if !placement.worker_groups.is_empty() {
        return Deployment::read(&placement.worker_groups)?
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
    Ok(WorkerGroupConfig::data_parallel(
        &hosts,
        &placement.device,
        placement.data_parallel_size.max(1) as usize,
        tensor_parallel_size,
        placement.queue_depth.max(1) as usize,
        WorkerGroupConfig::single_component(DEFAULT_COMPONENT, tensor_parallel_size),
    ))
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
            segments: Vec::new(),
            retire: Vec::new(),
            speculation: None,
            profiling: None,
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
}

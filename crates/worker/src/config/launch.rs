//! Resolve a launch descriptor once, before model loading and collectives.
//!
//! [`resolve`] reads every descriptor value into the immutable
//! [`WorkerLaunch`] the rank runs with, checking only what the rank alone
//! knows. All deployment decisions are made here; the numerical library
//! receives only resolved values.

use std::collections::{BTreeMap, HashSet};

use uniserve_core::ComponentConfig;
use uniserve_ipc::LaunchDescriptor;

use super::device::{Device, generation_device};
use super::{
    DEFAULT_FLOW_GRAPH_BATCH_SIZES, DEFAULT_FLOW_GRAPH_SHAPES, DescriptorError, WorkerConfig,
    default_decode_sizes, default_prefill_sizes,
};

/// Channel and product-publication settings of the rank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IpcConfig {
    pub acknowledgment_slot: u32,
    pub host_slots: Vec<u32>,
    pub products_cross_hosts: bool,
    /// Most unresolved requests on the channel.
    pub queue_depth: usize,
    /// How long one job may run on a lane before the rank reports the lane
    /// stalled: half the head's watchdog timeout.
    pub stall_bound: std::time::Duration,
}

/// Where a group's ranks meet to form their host-collective mesh.
///
/// Only the group's first rank holds `listen_fd`: the launcher bound the
/// listening socket before the rank existed and the rank inherited it, so
/// the port is never released between reservation and use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RendezvousConfig {
    /// `host:port` every member connects to.
    pub address: String,
    /// Inherited listening descriptor number (first rank only).
    pub listen_fd: Option<i32>,
}

/// This rank's membership in the deployment's expert-parallel union.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertUnion {
    pub rank: usize,
    pub size: usize,
    /// Leading union ranks that run attention; zero when experts are
    /// colocated with attention.
    pub attention_ranks: usize,
    pub rendezvous: RendezvousConfig,
    pub exchange: String,
}

impl ExpertUnion {
    /// Experts run on dedicated ranks separate from attention.
    pub(crate) fn disaggregated(&self) -> bool {
        self.attention_ranks > 0
    }
}

/// The checkpoint a model rank loads.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSource {
    pub path: String,
    pub base_model: Option<String>,
    /// Precision selectors as a JSON object text, interpreted by the
    /// numerical library's rank loading (`uniserve.loading.rank`).
    pub quantization: String,
}

/// Checkpoint reading options, interpreted by the numerical library.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadOptions {
    /// `auto`, `safetensors` or `pt`.
    pub format: String,
    /// `eager`, `dummy` or `layered`.
    pub mode: String,
    pub download_dir: Option<String>,
    pub num_threads: Option<u32>,
}

/// FlashInfer attention options passed to the attention backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttentionOptions {
    /// Shared workspace bytes.
    pub workspace_size: u64,
    pub decode_backend: String,
    pub prefill_backend: String,
    pub disable_split_kv: bool,
    pub decode_split_tile_size: Option<u32>,
    pub prefill_split_tile_size: Option<u32>,
    pub use_tensor_core: Option<bool>,
}

/// Physical transfer mechanisms this rank binds and exports through.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataPlane {
    pub backends: Vec<String>,
    pub exports: Vec<String>,
}

/// Everything a rank runs with, resolved from its launch descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkerLaunch {
    pub ipc: IpcConfig,
    /// Rank index among the worker's ranks on this host.
    pub local_rank: usize,
    /// Torch backend override for numerical process groups.
    pub distributed_backend: Option<String>,
    /// Group rendezvous; present for a multi-rank group outside an expert union.
    pub rendezvous: Option<RendezvousConfig>,
    pub expert_parallel: Option<ExpertUnion>,
    /// `None` for the stub model.
    pub model: Option<ModelSource>,
    pub data_plane: DataPlane,
    pub execution: WorkerConfig,
    pub attention: AttentionOptions,
    pub load: LoadOptions,
    /// Load the deterministic stub model.
    pub stub: bool,
    /// Placed components by name; empty for an expert-only rank.
    pub components: BTreeMap<String, ComponentConfig>,
}

impl WorkerLaunch {
    /// The rank's primary device.
    pub fn device(&self) -> Device {
        self.execution.device
    }
}

/// Resolve a launch descriptor.
///
/// The command line and the engine validate the launch values they own
/// before they write the descriptor (`--lane` validates each lane where
/// the operator states it), so this checks only what the rank alone knows:
/// its device, the transports it can bind, the precisions its device
/// computes, and the text values it parses (`--worker-mesh`, the graph
/// catalogs, the lanes taken together). Fails with the first violated
/// rule; messages name the launch option the operator sets, because the
/// engine reports them as the rank's exit cause.
pub fn resolve(descriptor: LaunchDescriptor) -> Result<WorkerLaunch, DescriptorError> {
    let role = descriptor.role.trim().to_owned();
    let experts = expert_parallel(&descriptor)?;

    let device = Device::parse(&descriptor.device, "--device")?;
    let generation = generation_device(descriptor.mesh.as_deref().unwrap_or_default(), device)?;
    let backends = transfer_backends(&descriptor.transfer_backends)?;
    let exports = transfer_backends(&descriptor.export_backends)?;
    if exports.iter().any(|name| !backends.contains(name)) {
        return Err(DescriptorError::new(
            "export backends must be bound transports",
        ));
    }
    if backends.iter().any(|name| name == "cuda_vmm") && !device.is_cuda() {
        return Err(DescriptorError::new(
            "CUDA VMM requires a CUDA worker device",
        ));
    }

    let stub = descriptor.no_model;
    let model = text(Some(&descriptor.model));
    if !stub && model.is_none() {
        return Err(DescriptorError::new(
            "--model is required for a model worker",
        ));
    }

    let rendezvous = rendezvous(&descriptor);
    let load = load_options(&descriptor)?;
    let attention = attention_options(&descriptor)?;
    let execution = execution(&descriptor, device, generation, &role, experts.as_ref())?;

    Ok(WorkerLaunch {
        ipc: IpcConfig {
            acknowledgment_slot: descriptor.acknowledgment_slot,
            host_slots: descriptor.host_slots.clone(),
            products_cross_hosts: descriptor.products_cross_hosts,
            queue_depth: descriptor.queue_depth,
            // Half the head's watchdog, so a stalled lane is reported, and
            // named in its requests' failure, before the head replaces the
            // group, even for a job that started up to half of it after the
            // group's last progress.
            stall_bound: std::time::Duration::from_millis(descriptor.watchdog_timeout_ms / 2),
        },
        local_rank: descriptor.local_rank as usize,
        distributed_backend: text(descriptor.distributed_backend.as_deref()),
        rendezvous,
        expert_parallel: experts,
        model: model.map(|path| ModelSource {
            path,
            base_model: text(descriptor.base_model.as_deref()),
            quantization: descriptor.quantization_config.to_string(),
        }),
        data_plane: DataPlane { backends, exports },
        execution,
        attention,
        load,
        stub,
        components: descriptor.components,
    })
}

/// A trimmed, non-empty text value.
fn text(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn transfer_backends(value: &str) -> Result<Vec<String>, DescriptorError> {
    let names: Vec<String> = value
        .split(',')
        .map(|name| name.trim().to_owned())
        .collect();
    if names.iter().collect::<HashSet<_>>().len() != names.len() {
        return Err(DescriptorError::new(
            "transfer backends must be nonempty and unique",
        ));
    }
    if names.iter().any(|name| {
        !matches!(
            name.as_str(),
            "local" | "posix_shm" | "cuda_vmm" | "channel"
        )
    }) {
        return Err(DescriptorError::new(
            "transfer backends must name local, posix_shm, cuda_vmm, or channel",
        ));
    }
    Ok(names)
}

/// Where this rank's group forms its host-collective mesh.
///
/// The process that spawns the group's first rank, the engine or the
/// launcher of that rank's host, binds the rendezvous socket before the
/// rank exists and names it in that rank's descriptor alone.
fn rendezvous(descriptor: &LaunchDescriptor) -> Option<RendezvousConfig> {
    text(descriptor.rendezvous_address.as_deref()).map(|address| RendezvousConfig {
        address,
        listen_fd: descriptor.rendezvous_listen_fd,
    })
}

/// This rank's place in the expert-parallel union, when the launch names
/// one. The engine places the union and validates its shape before it
/// launches a rank.
fn expert_parallel(descriptor: &LaunchDescriptor) -> Result<Option<ExpertUnion>, DescriptorError> {
    let Some(value) = &descriptor.expert_parallel else {
        return Ok(None);
    };
    let address = text(value.address.as_deref()).ok_or_else(|| {
        DescriptorError::new("an expert-parallel union requires its store address")
    })?;
    Ok(Some(ExpertUnion {
        rank: value.rank as usize,
        size: value.size as usize,
        attention_ranks: value.attention_ranks as usize,
        rendezvous: RendezvousConfig {
            address,
            listen_fd: value.listen_fd,
        },
        exchange: value.exchange.clone(),
    }))
}

/// Separate the descriptor's one checkpoint reader selector into a file
/// format and a loading mode: a file format loads eagerly; `dummy` and
/// `layered` select a mode and leave the format to detection.
fn load_options(descriptor: &LaunchDescriptor) -> Result<LoadOptions, DescriptorError> {
    let selected = descriptor.load_format.as_str();
    let (format, mode) = match selected {
        "auto" | "safetensors" | "pt" => (selected, "eager"),
        "dummy" | "layered" => ("auto", selected),
        _ => {
            return Err(DescriptorError::new(format!(
                "unknown checkpoint load format {selected:?}"
            )));
        }
    };
    Ok(LoadOptions {
        format: format.to_owned(),
        mode: mode.to_owned(),
        download_dir: text(descriptor.download_dir.as_deref()),
        num_threads: descriptor.load_threads,
    })
}

fn attention_options(descriptor: &LaunchDescriptor) -> Result<AttentionOptions, DescriptorError> {
    let use_tensor_core = descriptor
        .flashinfer_use_tensor_core
        .as_deref()
        .map(|value| match value.trim().to_lowercase().as_str() {
            "true" => Ok(true),
            "false" => Ok(false),
            _ => Err(DescriptorError::new("expected an optional bool token")),
        })
        .transpose()?;
    Ok(AttentionOptions {
        workspace_size: descriptor.flashinfer_workspace_size,
        decode_backend: descriptor.flashinfer_decode_backend.clone(),
        prefill_backend: descriptor.flashinfer_prefill_backend.clone(),
        disable_split_kv: descriptor.flashinfer_disable_split_kv,
        decode_split_tile_size: descriptor.flashinfer_decode_split_tile_size,
        prefill_split_tile_size: descriptor.flashinfer_prefill_split_tile_size,
        use_tensor_core,
    })
}

/// The execution settings: every launch value as the descriptor states it,
/// and the rank's own graph catalogs where the descriptor leaves them unset.
fn execution(
    descriptor: &LaunchDescriptor,
    device: Device,
    generation: Option<Device>,
    role: &str,
    experts: Option<&ExpertUnion>,
) -> Result<WorkerConfig, DescriptorError> {
    let expert_exchange =
        experts.map_or_else(|| "alltoall".to_owned(), |union| union.exchange.clone());
    let mut config = WorkerConfig {
        device,
        generation_device: generation,
        rank: descriptor.rank as usize,
        world_size: descriptor.world_size as usize,
        role: role.to_owned(),
        expert_exchange,
        expert_microbatches: descriptor.expert_microbatches as usize,
        block_size: descriptor.block_size.map(|size| size as usize),
        max_total_tokens: descriptor.max_total_tokens.map(|tokens| tokens as usize),
        attention_backend: descriptor.attention_backend.clone(),
        max_batch_calls: descriptor.max_batch_calls as usize,
        max_batch_tokens: descriptor.max_batch_tokens as usize,
        max_model_len: descriptor.max_model_len as usize,
        max_video_seconds: descriptor.max_video_seconds,
        max_condition_rows: descriptor.max_condition_rows,
        ffmpeg: descriptor.ffmpeg.clone(),
        min_video_seconds: descriptor.min_video_seconds,
        video_text_capacities: Vec::new(),
        video_canvases: descriptor.video_canvases.clone(),
        deployment_components: descriptor.deployment_components.clone(),
        canvas_sampling: None,
        max_request_pool_size: descriptor.max_request_pool_size as usize,
        encoder_cache_entries: descriptor.encoder_cache_entries as usize,
        pool_storage_bytes: None,
        model_dtype: descriptor.model_dtype.clone(),
        prefix_cache_dtype: descriptor.prefix_cache_dtype.clone(),
        mem_fraction_static: descriptor.mem_fraction_static,
        lanes: descriptor.lane.clone(),
        graph_policy: descriptor.graph_policy.clone(),
        decode_graph_batch_sizes: default_decode_sizes(),
        prefill_cuda_graph: descriptor.prefill_cuda_graph,
        prefill_outputs: descriptor.prefill_outputs,
        prefill_graph_token_sizes: default_prefill_sizes(),
        flow_cuda_graph: descriptor.flow_cuda_graph,
        flow_graph_batch_sizes: DEFAULT_FLOW_GRAPH_BATCH_SIZES.to_vec(),
        flow_graph_shapes: DEFAULT_FLOW_GRAPH_SHAPES.to_vec(),
    };

    if config.expert_exchange == "alltoall" && experts.is_some() {
        // Each local graph also captures larger transfer capacities.
        // Geometric small buckets, then 512-token spacing, bound that
        // product.
        config.prefill_graph_token_sizes = [4, 8, 16, 32, 64, 128, 256, 512]
            .into_iter()
            .chain((1024..=16384).step_by(512))
            .collect();
    }

    for (value, buckets) in [
        (
            &descriptor.decode_graph_batch_sizes,
            &mut config.decode_graph_batch_sizes,
        ),
        (
            &descriptor.prefill_graph_token_sizes,
            &mut config.prefill_graph_token_sizes,
        ),
        (
            &descriptor.flow_graph_batch_sizes,
            &mut config.flow_graph_batch_sizes,
        ),
        (
            &descriptor.video_text_capacities,
            &mut config.video_text_capacities,
        ),
    ] {
        if let Some(value) = value {
            *buckets = positive_csv(value)?;
        }
    }
    if let Some(value) = &descriptor.flow_graph_shapes {
        config.flow_graph_shapes.clear();
        for shape in value.split(',') {
            let shape = shape.trim().to_lowercase();
            let (height, width) = shape
                .split_once('x')
                .ok_or_else(|| DescriptorError::new("flow graph shapes must use HEIGHTxWIDTH"))?;
            let shape = (positive_number(height)?, positive_number(width)?);
            if config.flow_graph_shapes.contains(&shape) {
                return Err(DescriptorError::new(
                    "flow graph shapes must be positive and unique",
                ));
            }
            config.flow_graph_shapes.push(shape);
        }
    }
    if let Some(sampling) = descriptor.canvas_sampling {
        if let Some(parameter) = sampling.invalid_parameter() {
            return Err(DescriptorError::new(format!(
                "invalid canvas sampling {parameter}"
            )));
        }
        config.canvas_sampling = Some(sampling);
    }
    config.validate().map_err(DescriptorError::from)?;
    Ok(config)
}

fn positive_number(value: &str) -> Result<usize, DescriptorError> {
    value
        .trim()
        .parse::<usize>()
        .ok()
        .filter(|&value| value > 0)
        .ok_or_else(|| DescriptorError::new("bucket sizes must be positive integers"))
}

fn positive_csv(value: &str) -> Result<Vec<usize>, DescriptorError> {
    let values = value
        .split(',')
        .map(positive_number)
        .collect::<Result<Vec<_>, _>>()?;
    if values.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(DescriptorError::new(
            "integer bucket lists must be strictly increasing",
        ));
    }
    Ok(values)
}

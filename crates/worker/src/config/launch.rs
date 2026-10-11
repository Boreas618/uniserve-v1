//! Resolve a launch descriptor once, before model loading and collectives.
//!
//! [`resolve`] validates every descriptor value and produces the immutable
//! [`WorkerLaunch`] the rank runs with. All deployment decisions are made
//! here; the numerical library receives only resolved values.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use uniserve_core::ComponentConfig;
use uniserve_ipc::{CallKind, ForwardMode, LaunchDescriptor, MediaCall, TransferMode};

use super::device::{Device, generation_device};
use super::{DescriptorError, WorkerConfig};

/// Channel and product-publication settings of the rank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IpcConfig {
    pub acknowledgment_slot: u32,
    pub host_slots: Vec<u32>,
    pub products_cross_hosts: bool,
    /// Most unresolved requests on the channel.
    pub queue_depth: usize,
    /// Largest frame payload, in bytes.
    pub max_payload_bytes: usize,
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
    pub fn disaggregated(&self) -> bool {
        self.attention_ranks > 0
    }
}

/// The checkpoint a model rank loads.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSource {
    pub path: String,
    pub base_model: Option<String>,
    /// Precision selectors as a JSON object text, interpreted by the
    /// numerical library's loading interface.
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
    pub checksum_manifest: Option<String>,
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
    pub worker_id: String,
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
    /// Call kinds the deployment allows this rank to serve.
    pub supported_calls: BTreeSet<CallKind>,
}

impl WorkerLaunch {
    /// The rank's primary device.
    pub fn device(&self) -> Device {
        self.execution.device
    }
}

/// Capability groups selectable with `--supported-calls`, and their kinds.
const CAPABILITY_GROUPS: &[(&str, &[CallKind])] = {
    use CallKind::{Forward, Media, Transfer};
    &[
        ("ar_extend", &[Forward(ForwardMode::Prefill)]),
        ("ar_decode", &[Forward(ForwardMode::Decode)]),
        ("ar_verify", &[Forward(ForwardMode::Verify)]),
        ("token_denoising", &[Forward(ForwardMode::TokenDenoising)]),
        ("encoder_vision", &[Media(MediaCall::VisionEncoding)]),
        ("encoder_latent", &[Media(MediaCall::LatentEncoding)]),
        ("encoder_text", &[Media(MediaCall::TextEncoding)]),
        ("transfer_product", &[Transfer(TransferMode::Tensor)]),
        ("transfer_kv_export", &[Transfer(TransferMode::KvExport)]),
        ("transfer_kv_install", &[Transfer(TransferMode::KvInstall)]),
        ("diffusion_prepare", &[Media(MediaCall::LatentPreparation)]),
        ("diffusion_step", &[Media(MediaCall::Denoising)]),
        (
            "diffusion_finalize",
            &[Media(MediaCall::ImageDecoding), Media(MediaCall::Muxing)],
        ),
        (
            "diffusion_decode",
            &[
                Media(MediaCall::VideoDecoding),
                Media(MediaCall::AudioDecoding),
            ],
        ),
        ("media_read", &[Media(MediaCall::MediaReading)]),
        (
            "media_append",
            &[
                Media(MediaCall::VideoEncoding),
                Media(MediaCall::AudioEncoding),
            ],
        ),
    ]
};

/// Validate and resolve a launch descriptor.
///
/// Fails with the first violated rule. Messages name the launch option the
/// operator sets, because the engine reports them as the rank's exit cause.
pub fn resolve(descriptor: LaunchDescriptor) -> Result<WorkerLaunch, DescriptorError> {
    let role = descriptor.role.trim().to_owned();
    let experts = expert_parallel(&descriptor, &role)?;
    let supported_calls = if role == "experts" {
        if experts
            .as_ref()
            .is_none_or(|union| !union.disaggregated() || union.rank < union.attention_ranks)
        {
            return Err(DescriptorError::new(
                "expert workers require a disaggregated expert union",
            ));
        }
        BTreeSet::new()
    } else {
        supported_calls(descriptor.supported_calls.as_deref())?
    };

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

    let quantization = match &descriptor.quantization_config {
        serde_json::Value::Object(fields) => fields.clone(),
        _ => {
            return Err(DescriptorError::new(
                "--quantization-config must be a JSON object",
            ));
        }
    };
    // A KV dtype in the precision selectors overrides `--kv-cache-dtype`, so
    // capacity planning and loading see one resolved value.
    let kv_override = match quantization.get("kv_cache_dtype") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(dtype)) => Some(dtype.clone()),
        Some(_) => return Err(DescriptorError::new("worker KV dtype is unsupported")),
    };
    if device.is_cuda() || generation.is_some_and(Device::is_cuda) {
        cuda_representation(&descriptor, kv_override.as_deref())?;
    }

    if descriptor.queue_depth == 0 {
        return Err(DescriptorError::new("--queue-depth must be positive"));
    }
    if descriptor.ipc_payload_cap == 0 {
        return Err(DescriptorError::new("--ipc-payload-cap must be positive"));
    }
    let stub = descriptor.no_model;
    let model = text(Some(&descriptor.model));
    if stub && !descriptor.allow_stub {
        return Err(DescriptorError::new(
            "--no-model loads synthetic outputs and requires --allow-stub",
        ));
    }
    if !stub && model.is_none() {
        return Err(DescriptorError::new(
            "--model is required for a model worker",
        ));
    }

    let rendezvous = rendezvous(&descriptor)?;
    let load = load_options(&descriptor)?;
    let attention = attention_options(&descriptor)?;
    let components = components(&descriptor, &role)?;
    let mut execution = execution(&descriptor, device, generation, &role, experts.as_ref())?;
    if kv_override.is_some() {
        execution.kv_cache_dtype = kv_override;
        execution.validate().map_err(DescriptorError::from)?;
    }

    Ok(WorkerLaunch {
        worker_id: descriptor.worker_id.clone(),
        ipc: IpcConfig {
            acknowledgment_slot: descriptor.acknowledgment_slot,
            host_slots: descriptor.host_slots.clone(),
            products_cross_hosts: descriptor.products_cross_hosts,
            queue_depth: descriptor.queue_depth,
            max_payload_bytes: descriptor.ipc_payload_cap,
        },
        local_rank: descriptor.local_rank as usize,
        distributed_backend: text(descriptor.distributed_backend.as_deref()),
        rendezvous,
        expert_parallel: experts,
        model: model.map(|path| ModelSource {
            path,
            base_model: text(descriptor.base_model.as_deref()),
            quantization: serde_json::Value::Object(quantization).to_string(),
        }),
        data_plane: DataPlane { backends, exports },
        execution,
        attention,
        load,
        stub,
        components,
        supported_calls,
    })
}

/// A trimmed, non-empty text value.
fn text(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn supported_calls(value: Option<&str>) -> Result<BTreeSet<CallKind>, DescriptorError> {
    let all = CAPABILITY_GROUPS
        .iter()
        .map(|(name, _)| *name)
        .collect::<Vec<_>>()
        .join(",");
    let selected = value.map_or(all, str::to_owned);
    let mut seen = HashSet::new();
    let mut calls = BTreeSet::new();
    for name in selected
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
    {
        if !seen.insert(name) {
            return Err(DescriptorError::new(
                "supported calls contain duplicate entries",
            ));
        }
        let (_, kinds) = CAPABILITY_GROUPS
            .iter()
            .find(|(group, _)| *group == name)
            .ok_or_else(|| {
                DescriptorError::new(format!("unknown call in supported calls {selected:?}"))
            })?;
        calls.extend(kinds.iter().copied());
    }
    if calls.is_empty() {
        return Err(DescriptorError::new(
            "supported calls must list at least one group",
        ));
    }
    Ok(calls)
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
    if names
        .iter()
        .any(|name| !matches!(name.as_str(), "local" | "shm" | "cuda_vmm" | "channel"))
    {
        return Err(DescriptorError::new(
            "transfer backends must name local, shm, cuda_vmm, or channel",
        ));
    }
    Ok(names)
}

/// Refuse precisions no native CUDA attention kernel computes.
fn cuda_representation(
    descriptor: &LaunchDescriptor,
    kv_override: Option<&str>,
) -> Result<(), DescriptorError> {
    if descriptor.model_dtype == "float32" {
        return Err(DescriptorError::new(
            "--dtype float32 has no native CUDA attention kernel: CUDA attention computes in bfloat16 or float16",
        ));
    }
    let dtype = kv_override.or(descriptor.kv_cache_dtype.as_deref());
    if dtype == Some("float8_e4m3fn") {
        let option = if kv_override.is_some() {
            "--quantization-config kv_cache_dtype"
        } else {
            "--kv-cache-dtype"
        };
        return Err(DescriptorError::new(format!(
            "{option} float8_e4m3fn has no native CUDA attention kernel: the cache keeps one FP8 scale per block, which no native kernel reads (the FP8-KV TensorRT-LLM kernels take FP8 queries with per-tensor BMM1/BMM2 scales); use a bfloat16 or float16 KV cache"
        )));
    }
    Ok(())
}

fn rendezvous(descriptor: &LaunchDescriptor) -> Result<Option<RendezvousConfig>, DescriptorError> {
    let address = text(descriptor.rendezvous_address.as_deref());
    let fd = descriptor.rendezvous_listen_fd;
    let Some(address) = address else {
        if fd.is_some() {
            return Err(DescriptorError::new(
                "a rendezvous socket requires its address",
            ));
        }
        return Ok(None);
    };
    let first = descriptor.rank == 0;
    if first != fd.is_some() {
        return Err(DescriptorError::new(if first {
            "the first rank must inherit its rendezvous socket"
        } else {
            "only the first rank inherits a rendezvous socket"
        }));
    }
    store_address(&address)?;
    Ok(Some(RendezvousConfig {
        address,
        listen_fd: fd,
    }))
}

/// Check that a rendezvous address names a host and a port.
fn store_address(address: &str) -> Result<(), DescriptorError> {
    let (_, port) = address
        .rsplit_once(':')
        .ok_or_else(|| DescriptorError::new("store address must be host:port"))?;
    port.parse::<u16>()
        .map_err(|error| DescriptorError::new(format!("invalid store port: {error}")))?;
    Ok(())
}

/// Validate the expert-parallel union membership.
fn expert_parallel(
    descriptor: &LaunchDescriptor,
    role: &str,
) -> Result<Option<ExpertUnion>, DescriptorError> {
    let Some(value) = &descriptor.expert_parallel else {
        return Ok(None);
    };
    let (rank, size, attention) = (
        i64::from(value.rank),
        i64::from(value.size),
        i64::from(value.attention_ranks),
    );
    let exchange = value.exchange.as_str();
    if !matches!(exchange, "alltoall" | "megamoe" | "dwdp" | "deepep") {
        return Err(DescriptorError::new(format!(
            "unknown expert exchange {exchange:?}; expected alltoall, megamoe, dwdp or deepep"
        )));
    }
    if size < 2 {
        return Err(DescriptorError::new(
            "an expert-parallel world spans at least two ranks",
        ));
    }
    if rank >= size {
        return Err(DescriptorError::new(
            "expert-parallel rank must satisfy 0 <= rank < size",
        ));
    }
    if attention >= size {
        return Err(DescriptorError::new(
            "attention ranks must leave at least one expert rank",
        ));
    }
    let start = rank - i64::from(descriptor.rank);
    let world_size = i64::from(descriptor.world_size);
    let stop = start + world_size;
    if start < 0 || stop > size || (start < attention && attention < stop) {
        return Err(DescriptorError::new(
            "worker ranks must fit within one role in the expert union",
        ));
    }
    if attention != 0
        && (!matches!(exchange, "deepep" | "megamoe")
            || ((rank >= attention) != (role == "experts")))
    {
        return Err(DescriptorError::new(
            "disaggregated role or expert exchange disagrees with placement",
        ));
    }
    if attention == 0 && (world_size != 1 || exchange == "deepep") {
        return Err(DescriptorError::new(
            "an expert-parallel replica joins its world with its only rank",
        ));
    }
    let address = text(value.address.as_deref())
        .ok_or_else(|| DescriptorError::new("store address must be host:port"))?;
    store_address(&address)?;
    if (rank == 0) != value.listen_fd.is_some() {
        return Err(DescriptorError::new(
            "exactly the expert-parallel world's rank 0 serves its store",
        ));
    }
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
        checksum_manifest: text(descriptor.checksum_manifest.as_deref()),
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
    if descriptor.flashinfer_workspace_size == 0 {
        return Err(DescriptorError::new(
            "FlashInfer workspace size must be positive",
        ));
    }
    for (name, tile) in [
        ("decode", descriptor.flashinfer_decode_split_tile_size),
        ("prefill", descriptor.flashinfer_prefill_split_tile_size),
    ] {
        if tile == Some(0) {
            return Err(DescriptorError::new(format!(
                "FlashInfer {name} split tile size must be positive"
            )));
        }
    }
    for backend in [
        &descriptor.flashinfer_decode_backend,
        &descriptor.flashinfer_prefill_backend,
    ] {
        if !matches!(backend.as_str(), "auto" | "fa2" | "fa3") {
            return Err(DescriptorError::new(format!(
                "unsupported FlashInfer backend {backend:?}"
            )));
        }
    }
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

/// Resolve rank placement once, before any collective begins.
fn components(
    descriptor: &LaunchDescriptor,
    role: &str,
) -> Result<BTreeMap<String, ComponentConfig>, DescriptorError> {
    if role == "experts" {
        if !descriptor.components.is_empty() {
            return Err(DescriptorError::new(
                "expert workers have no request components",
            ));
        }
        return Ok(BTreeMap::new());
    }
    if descriptor.components.is_empty() {
        return Err(DescriptorError::new(
            "component configuration must not be empty",
        ));
    }
    for (name, component) in &descriptor.components {
        component
            .validate()
            .map_err(|error| DescriptorError::new(error.to_string()))?;
        if name.is_empty()
            || component
                .ranks
                .iter()
                .any(|&rank| rank >= descriptor.world_size as usize)
        {
            return Err(DescriptorError::new(
                "component contains ranks outside the process world or an empty name",
            ));
        }
    }
    Ok(descriptor.components.clone())
}

/// The execution settings, including bucket lists and lanes.
fn execution(
    descriptor: &LaunchDescriptor,
    device: Device,
    generation: Option<Device>,
    role: &str,
    experts: Option<&ExpertUnion>,
) -> Result<WorkerConfig, DescriptorError> {
    let positive = |value: Option<u64>| -> Result<Option<usize>, DescriptorError> {
        match value {
            Some(0) => Err(DescriptorError::new(
                "optional integer tuning values must be positive when set",
            )),
            value => Ok(value.map(|value| value as usize)),
        }
    };
    let kv_cache_dtype = descriptor
        .kv_cache_dtype
        .as_deref()
        .map(|dtype| {
            let dtype = dtype.trim();
            if dtype.is_empty() {
                return Err(DescriptorError::new(
                    "an explicitly provided string setting must not be empty",
                ));
            }
            Ok(dtype.to_owned())
        })
        .transpose()?;

    let mut config = WorkerConfig {
        device,
        generation_device: generation,
        rank: descriptor.rank as usize,
        world_size: descriptor.world_size as usize,
        role: if role.is_empty() {
            "model".into()
        } else {
            role.to_owned()
        },
        expert_microbatches: descriptor.expert_microbatches as usize,
        max_batch_calls: descriptor.max_batch_calls as usize,
        max_batch_tokens: descriptor.max_batch_tokens as usize,
        max_request_pool_size: descriptor.max_request_pool_size as usize,
        max_sequence_tokens: descriptor.max_model_len as usize,
        max_video_seconds: descriptor.max_video_seconds,
        max_condition_rows: descriptor.max_condition_rows as usize,
        encoder_cache_entries: descriptor.encoder_cache_entries as usize,
        ffmpeg: descriptor.ffmpeg.clone(),
        min_video_seconds: descriptor.min_video_seconds,
        video_canvases: descriptor.video_canvases.clone(),
        deployment_components: descriptor.deployment_components.clone(),
        model_dtype: descriptor.model_dtype.clone(),
        kv_storage_fraction: descriptor.kv_memory_fraction,
        prefill_cuda_graph: descriptor.prefill_cuda_graph,
        prefill_outputs: descriptor.prefill_outputs,
        flow_cuda_graph: descriptor.flow_cuda_graph,
        graph_policy: descriptor.graph_policy.clone(),
        attention_backend: Some(descriptor.attention_backend.clone()),
        block_size: positive(descriptor.block_size.map(u64::from))?,
        kv_token_capacity: positive(descriptor.kv_token_capacity)?,
        kv_cache_dtype,
        lanes: descriptor.lane.clone(),
        ..WorkerConfig::default()
    };

    if let Some(experts) = experts {
        config.expert_exchange = experts.exchange.clone();
        if config.expert_microbatches > 1 && !experts.disaggregated() {
            return Err(DescriptorError::new(
                "expert microbatches require disaggregated experts",
            ));
        }
        if config.expert_exchange == "alltoall" {
            // Each local graph also captures larger transfer capacities.
            // Geometric small buckets bound that product; the largest gap
            // remains within the ordinary default KV padding reservation.
            config.prefill_graph_token_sizes = [4, 8, 16, 32, 64, 128, 256, 512]
                .into_iter()
                .chain((1024..=16384).step_by(512))
                .collect();
        }
    } else if config.expert_microbatches > 1 {
        return Err(DescriptorError::new(
            "expert microbatches require disaggregated experts",
        ));
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
    for lane in &config.lanes {
        lane.validate()
            .map_err(|error| DescriptorError::new(error.to_string()))?;
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

    // The service uses a 24-fps clock; model windows may extend it by
    // sixteen frames. Keep the resolved count within the IPC frame index.
    let frames = (config.max_video_seconds * 24.0).round_ties_even();
    if !(1.0..=f64::from(u32::MAX - 16)).contains(&frames) {
        return Err(DescriptorError::new(
            "max-video-seconds must resolve to a supported frame count",
        ));
    }
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

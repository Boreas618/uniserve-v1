//! Startup of the image runtime: bounds, bindings and owned storage.
//!
//! The runtime serves every encoder, image denoiser and image decoder call
//! of a model that is not a video model. [`ImageRuntime::start`] prepares
//! and binds one component binding per call (at the call's module path, on
//! the stream serving its call kind), then binds the call's runner over
//! Rust-owned storage: encoders over the device's product arena, the image
//! denoiser over the latent pool's step workspace and the KV cache's slot
//! tables, the image decoder over the step workspace and its staging
//! bytes. The latent pool of image trajectories is allocated here, on the
//! denoiser's device.

use std::collections::HashMap;

use uniserve_cuda::{DeviceId, Stream};
use uniserve_ipc::{CallKind, MediaCall};

use super::ImageRuntime;
use super::latents::ImageLatents;
use crate::capacity::plan::LatentPlan;
use crate::config::{Device, WorkerConfig};
use crate::executor::forward::decodes_video;
use crate::executor::forward::token::start::TextBounds;
use crate::ffi::{DType, Device as StorageDevice, Handle};
use crate::host::HostLane;
use crate::model::LoadedModel;
use crate::model::bind::{self, Backends, BindingRequest, StreamWorkspaces, TextSize};
use crate::model::components::Capability;
use crate::model::image::{ImageDenoiserLayout, ImageModel, ImagePolicy};
use crate::model::interface::Interface;
use crate::model::registry::Components;
use crate::request::diffusion::Trajectories;
use crate::runner::GraphStorage;
use crate::runner::decoder::DecoderRunner;
use crate::runner::diffusion::image::{DiffusionCapacity, DiffusionStorage, ImageDiffusion};
use crate::runner::encoder::{EncoderConfig, EncoderInfo, EncoderRunner};
use crate::runner::graphs::reserve_pools;
use crate::runner::text::model_dtype;
use crate::storage::{LatentPool, LatentStorage, OutputPool, TensorStore};
use crate::worker::{Lanes, RankStorage};
use crate::{Error, Result};

/// Guidance branches one image evaluates at most per step.
const MAX_BRANCHES: usize = 3;

/// Prefix tokens of each scratch slot a captured denoising step reads; host
/// prefix lengths are rebound when the step replays.
pub(crate) const CAPTURE_PREFIX: usize = 16;

/// A denoising step captured at startup: `images` images of one size, each
/// evaluating `branches` guidance branches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FlowShape {
    pub(crate) images: usize,
    pub(crate) branches: usize,
    pub(crate) height: u32,
    pub(crate) width: u32,
}

/// The configured denoising steps that fit the denoiser's columns and the
/// latent pool, largest footprint first: every configured image size and
/// batch size with one to three guidance branches, keeping those within
/// the batch's calls, the evaluation's tokens, one image's sequence and the
/// pool's units.
fn flow_shapes(
    config: &WorkerConfig,
    layout: &ImageDenoiserLayout,
    max_calls: usize,
    tokens: usize,
    pool_units: usize,
) -> Vec<FlowShape> {
    let mut shapes = Vec::new();
    for &(height, width) in &config.flow_graph_shapes {
        let (height, width) = (height as u32, width as u32);
        let (Ok(units), Ok(length)) = (
            layout.latent_rows(height, width),
            layout.sequence_length(height, width),
        ) else {
            continue;
        };
        for &images in &config.flow_graph_batch_sizes {
            for branches in 1..=MAX_BRANCHES {
                if images > 0
                    && images <= max_calls
                    && images * length as usize * branches <= tokens
                    && units <= layout.max_sequence_tokens
                    && images * units as usize <= pool_units
                {
                    shapes.push(FlowShape {
                        images,
                        branches,
                        height,
                        width,
                    });
                }
            }
        }
    }
    shapes.sort_by_key(|shape| {
        std::cmp::Reverse(
            shape.images * shape.branches * shape.height as usize * shape.width as usize,
        )
    });
    shapes.dedup();
    shapes
}

/// Host workers encoding and preparing images.
const HOST_WORKERS: usize = 4;

/// What image startup borrows from the rank.
pub(crate) struct Startup<'a> {
    pub(crate) interface: &'a Interface,
    pub(crate) config: &'a WorkerConfig,
    pub(crate) request_slots: usize,
    pub(crate) queue_depth: usize,
    /// Host tasks the rank admits at once.
    pub(crate) host_capacity: usize,
    pub(crate) model: &'a LoadedModel,
    pub(crate) lanes: &'a Lanes,
    pub(crate) storage: &'a RankStorage,
    pub(crate) latent: Option<LatentPlan>,
    pub(crate) components: &'a mut Components,
    pub(crate) workspaces: &'a mut StreamWorkspaces,
    pub(crate) graphs: &'a mut GraphStorage<String, Handle>,
    /// The device's startup stream for resets.
    pub(crate) preparation: Option<&'a Stream>,
    /// The process group symmetric workspaces are allocated over.
    pub(crate) peers: Option<&'a bind::PeerGroup<'a>>,
}

/// One call the runtime binds.
struct Target {
    component: String,
    path: String,
    kind: CallKind,
    device: Device,
    stream: Option<Stream>,
}

/// The stream serving `kind` on `device`, borrowed from the lanes.
fn stream(lanes: &Lanes, kind: CallKind, device: Device) -> Result<Option<Stream>> {
    let Some(index) = device.cuda_index() else {
        return Ok(None);
    };
    let stream = lanes
        .stream_for(kind, DeviceId(index))
        .ok_or(Error::State("image calls have no execution stream"))?;
    // SAFETY: the lanes own the stream and close after the runtime.
    Ok(Some(unsafe {
        Stream::borrowed(DeviceId(index), stream.handle()?)
    }?))
}

fn borrow(stream: &Option<Stream>) -> Result<Option<Stream>> {
    stream
        .as_ref()
        // SAFETY: the lanes own the stream for the runtime's lifetime.
        .map(|stream| unsafe { Stream::borrowed(stream.device(), stream.raw()) })
        .transpose()
        .map_err(Error::from)
}

/// Tokens one denoising evaluation carries at most: every guidance branch
/// of the longest image sequence, or of a configured graph shape.
pub(crate) fn flow_tokens(layout: &ImageDenoiserLayout, config: &WorkerConfig) -> usize {
    let mut tokens = (layout.max_sequence_tokens + layout.framing_tokens) as usize;
    for &(height, width) in &config.flow_graph_shapes {
        if let Ok(length) = layout.sequence_length(height as u32, width as u32) {
            tokens = tokens.max(length as usize);
        }
    }
    // Denoising and prompt invocations share one token extent: a batch's
    // tokens beside one image's sequence.
    (tokens * MAX_BRANCHES).max(
        config.max_batch_tokens + (layout.max_sequence_tokens + layout.framing_tokens) as usize,
    )
}

/// The largest image a decoder stages: the longest latent sequence's
/// pixels.
fn max_image_pixels(layout: Option<&ImageDenoiserLayout>) -> usize {
    layout.map_or(1024 * 1024, |layout| {
        let side = layout.downsample as usize;
        layout.max_sequence_tokens as usize * side * side
    })
}

impl ImageRuntime {
    /// Create the runners of every image call on this rank, or `None` on
    /// a rank without one.
    pub(crate) fn start(startup: Startup<'_>) -> Result<Option<Self>> {
        let mut targets = Vec::new();
        for placed in startup.components.placed() {
            if placed.host()
                || decodes_video(
                    placed
                        .calls
                        .iter()
                        .flat_map(|call| call.capabilities.iter().copied()),
                )
            {
                continue;
            }
            for call in &placed.calls {
                for kind in call.kinds() {
                    let served = match kind {
                        CallKind::Media(MediaCall::VisionEncoding | MediaCall::LatentEncoding) => {
                            true
                        }
                        CallKind::Media(MediaCall::Denoising) => call.is(Capability::ImageDenoiser),
                        CallKind::Media(MediaCall::ImageDecoding) => true,
                        _ => false,
                    };
                    if served {
                        targets.push(Target {
                            component: placed.name.clone(),
                            path: call.path.clone(),
                            kind,
                            device: placed.device,
                            stream: stream(startup.lanes, kind, placed.device)?,
                        });
                    }
                }
            }
        }
        if targets.is_empty() {
            return Ok(None);
        }

        let interface = startup.interface;
        let config = startup.config;
        let loaded = startup.model.handle();
        let facts = ImageModel::describe(interface, loaded)?;
        let policy = ImagePolicy::new(interface, loaded.clone())?;
        let dtype = model_dtype(&config.model_dtype)?;
        let max_calls = config
            .max_batch_calls
            .min(startup.request_slots.max(1))
            .max(1);
        let encoder_config = EncoderConfig {
            max_images: max_calls,
            row_budget: config.max_batch_tokens,
            text_capacities: Vec::new(),
            max_text_tokens: config.max_sequence_tokens,
        };

        // The denoiser's attention geometry and the KV pool its context
        // reads.
        let kv = startup.storage.kv.as_ref();
        let denoising = targets
            .iter()
            .any(|target| target.kind == CallKind::Media(MediaCall::Denoising));
        let layout = facts.denoiser;
        let (bounds, capacity, pool) = match (denoising, kv, layout) {
            (false, ..) => (None, None, None),
            (true, Some(kv), Some(layout)) => {
                let bounds = TextBounds::new(
                    config,
                    startup.request_slots,
                    kv.plan(),
                    Some(kv.planes().num_units()),
                    config.primary_device().is_cuda(),
                )?;
                let tokens = flow_tokens(&layout, config);
                let width = bounds.widths.iter().copied().max().unwrap_or(1).max(1) as usize;
                let capacity = DiffusionCapacity {
                    rows: max_calls * MAX_BRANCHES,
                    tokens,
                    units: tokens,
                    tables: bounds.tables.len(),
                    width,
                    graph_widths: bounds.widths.iter().map(|&width| width as usize).collect(),
                };
                let views = kv
                    .views()?
                    .into_iter()
                    .try_fold(crate::ffi::Fields::new(), |fields, (name, view)| {
                        Ok::<_, Error>(fields.with(&name, view.to_tensor()?))
                    })?
                    .into_any();
                let pool = Handle::new(
                    interface
                        .function("cache.pool")?
                        .call_values(&[tvm_ffi::Any::from(kv.layout()?), views])?,
                );
                (Some(bounds), Some(capacity), Some(pool))
            }
            (true, None, _) => {
                return Err(Error::Invalid(
                    "an image denoiser requires a KV cache".into(),
                ));
            }
            (true, _, None) => {
                return Err(Error::Interface(
                    "an image denoiser reports no denoiser layout".into(),
                ));
            }
        };

        // Prepare every binding before binding any, so shared workspaces
        // hold the largest of them.
        let mut prepared = Vec::with_capacity(targets.len());
        let mut infos = Vec::with_capacity(targets.len());
        for target in &targets {
            let (size, info, cache) = match target.kind {
                CallKind::Media(MediaCall::VisionEncoding | MediaCall::LatentEncoding) => {
                    let info = EncoderInfo::describe(interface, loaded, &target.path, dtype)?;
                    (info.binding_size(&encoder_config), Some(info), None)
                }
                CallKind::Media(MediaCall::Denoising) => {
                    let capacity = capacity.as_ref().ok_or_else(|| {
                        Error::Invariant("an image denoiser lost its capacity".into())
                    })?;
                    (
                        Some(TextSize {
                            tokens: capacity.tokens,
                            rows: capacity.rows,
                        }),
                        None,
                        pool.clone(),
                    )
                }
                _ => (None, None, None),
            };
            infos.push(info);
            prepared.push(bind::prepare(
                interface,
                loaded,
                BindingRequest {
                    component: target.component.clone(),
                    path: target.path.clone(),
                    device: target.device,
                    stream: target.stream.as_ref().map_or(0, Stream::raw),
                    size,
                    backends: Backends::default(),
                    // The denoiser receives exact host lengths; encoders and
                    // decoders derive their own.
                    derive_host_lengths: cache.is_none(),
                    experts: None,
                    weights: None,
                    cache,
                },
            )?);
        }
        let bindings = bind::bind(interface, startup.workspaces, prepared, startup.peers)?;

        // The latent pool lives on the denoiser's device.
        let denoiser_device = targets
            .iter()
            .find(|target| target.kind == CallKind::Media(MediaCall::Denoising))
            .map(|target| target.device);
        let latents = match (denoiser_device, startup.latent) {
            (Some(device), Some(plan)) => {
                let width = u32::try_from(plan.latent_width)
                    .map_err(|_| Error::Invalid("latent width overflows".into()))?;
                let narrow = |value: usize, name: &str| {
                    u32::try_from(value).map_err(|_| Error::Invalid(format!("{name} overflows")))
                };
                let storage = LatentStorage::allocate(
                    narrow(plan.request_pool_size, "latent slots")?,
                    narrow(plan.num_pages, "latent pages")?,
                    narrow(plan.page_units, "latent page units")?,
                    width,
                    dtype,
                    true,
                    device.cuda_index(),
                    startup.preparation,
                )?;
                let pool =
                    LatentPool::new(plan.request_pool_size, plan.num_pages, plan.page_units)?;
                Some(ImageLatents::new(pool, storage))
            }
            (Some(_), None) => {
                return Err(Error::Invalid(
                    "an image denoiser requires a latent pool plan".into(),
                ));
            }
            (None, _) => None,
        };
        let samples = latents
            .as_ref()
            .map(|latents| latents.workspace().cloned())
            .transpose()?;

        let mut arenas = HashMap::new();
        let mut encoders = Vec::new();
        let mut encoder_routes = HashMap::new();
        let mut diffusion = None;
        let mut decoder = None;
        let mut units = None;
        for ((target, binding), info) in targets.iter().zip(&bindings).zip(infos) {
            let products = arena(&startup.storage.tensors, storage_device(target.device))?;
            arenas.insert(storage_device(target.device), products.clone());
            match target.kind {
                CallKind::Media(MediaCall::VisionEncoding | MediaCall::LatentEncoding) => {
                    let info = info.ok_or_else(|| {
                        Error::Invariant("an encoder binding lost its description".into())
                    })?;
                    let runner = EncoderRunner::bind(
                        interface,
                        binding,
                        loaded,
                        "",
                        info,
                        &encoder_config,
                        products,
                        borrow(&target.stream)?,
                    )?;
                    encoder_routes.insert((target.component.clone(), target.kind), encoders.len());
                    encoders.push(runner);
                }
                CallKind::Media(MediaCall::Denoising) => {
                    let (Some(layout), Some(capacity), Some(bounds), Some(kv), Some(latents)) = (
                        layout,
                        capacity.clone(),
                        bounds.as_ref(),
                        kv,
                        latents.as_ref(),
                    ) else {
                        return Err(Error::Invariant(
                            "an image denoiser lost its storage".into(),
                        ));
                    };
                    let samples = latents.workspace()?.clone();
                    let slot_tables = kv
                        .storage()
                        .views()
                        .into_iter()
                        .filter(|(name, _)| name != "verified_lengths" && name != "alloced_lens")
                        .collect();
                    let page_tokens: Vec<i64> = bounds
                        .tables
                        .iter()
                        .map(|table| i64::from(table.page_tokens))
                        .collect();
                    let windowed: Vec<i64> = bounds
                        .tables
                        .iter()
                        .map(|table| i64::from(table.windowed))
                        .collect();
                    let table_layout = crate::ffi::Fields::new()
                        .with("page_tokens", crate::ffi::integer_list(page_tokens))
                        .with("windowed", crate::ffi::integer_list(windowed))
                        .map();
                    let runner = ImageDiffusion::bind(
                        interface,
                        binding,
                        loaded,
                        "",
                        layout,
                        capacity,
                        DiffusionStorage {
                            samples,
                            slot_timesteps: latents.storage.timesteps().clone(),
                            products,
                            slot_tables,
                            layout: table_layout,
                        },
                        borrow(&target.stream)?,
                    )?;
                    units = Some(kv.bind_units(
                        &interface.function("cache.bind_units")?,
                        target.stream.as_ref(),
                    )?);
                    diffusion = Some((target.component.clone(), runner));
                }
                CallKind::Media(MediaCall::ImageDecoding) => {
                    let range = facts.decoder_range.ok_or_else(|| {
                        Error::Interface("an image decoder reports no pixel range".into())
                    })?;
                    let latents = samples.clone().ok_or_else(|| {
                        Error::Unsupported("image decoding requires an image denoiser".into())
                    })?;
                    let staging = max_image_pixels(layout.as_ref()) * 3 * max_calls;
                    let runner = DecoderRunner::bind(
                        interface,
                        binding,
                        loaded,
                        "",
                        range,
                        latents,
                        products,
                        staging,
                        borrow(&target.stream)?,
                    )?;
                    decoder = Some((target.component.clone(), runner));
                }
                _ => {}
            }
        }
        startup.components.add_bindings(bindings);

        // The configured denoising steps capture at warmup into one pool
        // per device the denoiser's modules run on: its own and a
        // generation tower's, reached through transfers.
        let mut flow = Vec::new();
        let mut flow_pools = Vec::new();
        if let (Some((_, runner)), Some(latents)) = (&diffusion, &latents)
            && config.graph_policy != "off"
            && config.flow_cuda_graph
            && let Some(ordinal) = runner.device().cuda_index()
        {
            flow = flow_shapes(
                config,
                runner.layout(),
                max_calls,
                runner.capacity().tokens,
                latents.pool.capacity_units(),
            );
            if !flow.is_empty() {
                let mut devices = vec![ordinal];
                devices.extend(
                    config
                        .devices()
                        .filter_map(|name| Device::parse(name, "device").ok())
                        .filter_map(Device::cuda_index)
                        .filter(|&device| device != ordinal),
                );
                flow_pools = reserve_pools(
                    startup.graphs,
                    &interface.function("graphs.pool")?,
                    runner.graph_key(),
                    "image:denoiser".to_owned(),
                    &devices,
                    None,
                )?;
            }
        }

        // Packed vision encoders capture their slot counts at warmup.
        let mut pools = HashMap::new();
        if config.graph_policy != "off" {
            let create_pool = interface.function("graphs.pool")?;
            for (index, runner) in encoders.iter().enumerate() {
                let (Some(key), Some(ordinal)) = (runner.graph_key(), runner.device().cuda_index())
                else {
                    continue;
                };
                let reserved = reserve_pools(
                    startup.graphs,
                    &create_pool,
                    key,
                    format!("image:encoder{index}"),
                    &[ordinal],
                    None,
                )?;
                if let Some(pool) = reserved.into_iter().next() {
                    pools.insert(index, pool);
                }
            }
        }

        // Completion words per call plus every image's pixels.
        let max_pixel_words = (max_image_pixels(layout.as_ref()) * 3).div_ceil(8) * max_calls;
        let device = config.primary_device().cuda_index();
        let outputs = OutputPool::new(
            device,
            startup.queue_depth.max(1) * 2,
            max_calls + max_pixel_words,
        );
        let host = HostLane::new(
            startup.host_capacity.max(HOST_WORKERS),
            HOST_WORKERS,
            "uniserve-image",
        )?;
        if let Some(stream) = startup.preparation {
            stream.synchronize()?;
        }
        Ok(Some(Self {
            facts,
            policy,
            encoders,
            encoder_routes,
            diffusion,
            decoder,
            latents,
            trajectories: Trajectories::default(),
            arenas,
            units,
            pools,
            flow,
            flow_pools,
            max_pixel_words,
            outputs,
            host,
        }))
    }
}

/// Where storage of a component device lives.
fn storage_device(device: Device) -> StorageDevice {
    match device {
        Device::Cuda(index) => StorageDevice::Cuda(index),
        Device::Cpu => StorageDevice::Cpu,
    }
}

/// The product arena of `device` as one `uint8` dimension.
fn arena(
    tensors: &TensorStore,
    device: StorageDevice,
) -> Result<crate::storage::memory::StorageView> {
    let view = tensors.arena(device)?;
    if view.dtype() != DType::U8 {
        return Err(Error::Invariant(
            "a product arena is bound as uint8 bytes".into(),
        ));
    }
    Ok(view)
}

/// Device bytes the image runtime allocates beside the KV and latent pools:
/// the denoiser's columns and the decoder's staging.
pub(crate) fn reserved_bytes(
    config: &WorkerConfig,
    request_slots: usize,
    layout: Option<&ImageDenoiserLayout>,
) -> u64 {
    let Some(layout) = layout else {
        return 0;
    };
    let calls = config.max_batch_calls.min(request_slots.max(1)).max(1);
    let tokens = flow_tokens(layout, config) as u64;
    let rows = (calls * MAX_BRANCHES) as u64;
    let predictions = tokens * u64::from(layout.width()) * layout.prediction_dtype.bytes() as u64;
    let columns = tokens * (3 * 8 + 8 * 4) + rows * 64;
    let staging = (max_image_pixels(Some(layout)) * 3 * calls) as u64;
    predictions + columns + staging
}

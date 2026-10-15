//! Startup of a rank's media runners: the runners of a video model's
//! placed components, and the runners of an image model's
//! encoders, denoiser and decoder.

use std::collections::HashMap;

use uniserve_cuda::{DeviceId, Stream};
use uniserve_ipc::{CallKind, MediaCall};

use crate::capacity::plan::LatentPlan;
use crate::config::placement::Capability;
use crate::config::placement::Components;
use crate::config::{Device, WorkerConfig};
use crate::cuda_graph::GraphStorage;
use crate::cuda_graph::reserve_pools;
use crate::executor::forward::decodes_video;
use crate::ffi::{Handle, Interpreter};
use crate::host::HostLane;
use crate::host::image::ImageTask;
use crate::lane::Lanes;
use crate::runner::ImageSetup;
use crate::runner::causal_lm::model_dtype;
use crate::runner::causal_lm::start::TextBounds;
use crate::runner::component::VideoSetup;
use crate::runner::decoder::DecoderRunner;
use crate::runner::decoder::runners::DecoderRunners;
use crate::runner::denoiser::image::{DiffusionCapacity, DiffusionStorage, ImageDiffusion};
use crate::runner::denoiser::image_model::{ImageDenoiserLayout, ImageModel, ImagePolicy};
use crate::runner::denoiser::runners::{DenoiserRunners, FlowShape};
use crate::runner::encoder::runners::EncoderRunners;
use crate::runner::encoder::{EncoderConfig, EncoderInfo, EncoderRunner};
use crate::runner::media::MediaRunner;
use crate::runner::scratch::Scratches;
use crate::storage::{ImageLatents, LatentPool, LatentStorage, OutputPool};
use crate::worker::LoadedModel;
use crate::worker::RankStorage;
use crate::{Error, Result};

/// Guidance branches one image evaluates at most per step.
const MAX_BRANCHES: usize = 3;

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
    pub(crate) config: &'a WorkerConfig,
    pub(crate) request_slots: usize,
    pub(crate) queue_depth: usize,
    /// Host tasks the rank admits at once.
    pub(crate) host_capacity: usize,
    pub(crate) model: &'a LoadedModel,
    pub(crate) lanes: &'a Lanes,
    pub(crate) storage: &'a RankStorage,
    pub(crate) latent: Option<LatentPlan>,
    pub(crate) components: &'a Components,
    /// The shared work areas of the rank's streams.
    pub(crate) scratches: &'a mut Scratches,
    pub(crate) graphs: &'a mut GraphStorage<String, Handle>,
    /// The device's startup stream for resets.
    pub(crate) preparation: Option<&'a Stream>,
}

/// One call startup binds.
struct Target {
    component: String,
    path: String,
    kind: CallKind,
    device: Device,
    stream: Option<Stream>,
}

/// The lane stream serving `kind` on `device`.
fn stream(lanes: &Lanes, kind: CallKind, device: Device) -> Result<Option<Stream>> {
    let Some(index) = device.cuda_index() else {
        return Ok(None);
    };
    let stream = lanes
        .stream_for(kind, DeviceId(index))
        .ok_or(Error::State("image calls have no execution stream"))?;
    Ok(Some(stream.stream()?.clone()))
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

/// One of an image model's runners on a rank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ImageRunner {
    /// The encoder at this index of the rank's image encoders.
    Encoder(usize),
    Denoiser,
    Decoder,
}

/// What serves an image model on this rank, as startup built it.
pub(crate) struct ImageOwners {
    pub(crate) model: ImageModel,
    pub(crate) policy: ImagePolicy,
    pub(crate) encoders: EncoderRunners,
    pub(crate) denoisers: DenoiserRunners,
    pub(crate) decoders: DecoderRunners,
    /// The latent pool of a rank holding an image denoiser.
    pub(crate) latents: Option<ImageLatents>,
    /// The runners with their names, in the order startup constructed
    /// them: the order their kernels are listed in; they close in reverse.
    pub(crate) order: Vec<(ImageRunner, String)>,
    /// The output leases of image batches.
    pub(crate) outputs: OutputPool,
    /// The lane preparing input images and encoding output images.
    pub(crate) host: HostLane<ImageTask>,
}

impl ImageOwners {
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

        let config = startup.config;
        let loaded = startup.model.handle();
        let facts = ImageModel::describe(loaded)?;
        let policy = ImagePolicy::new(loaded.clone())?;
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

        // The denoiser's attention geometry and the cache pool its context
        // reads.
        let prefix_cache = startup.storage.prefix_cache.as_ref();
        let denoising = targets
            .iter()
            .any(|target| target.kind == CallKind::Media(MediaCall::Denoising));
        let layout = facts.denoiser;
        let (bounds, capacity, pool) = match (denoising, prefix_cache, layout) {
            (false, ..) => (None, None, None),
            (true, Some(prefix_cache), Some(layout)) => {
                let bounds = TextBounds::new(
                    config,
                    startup.request_slots,
                    prefix_cache.plan(),
                    Some(prefix_cache.planes().num_units()),
                    config.device.is_cuda(),
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
                let pool = prefix_cache.pool()?;
                (Some(bounds), Some(capacity), Some(pool))
            }
            (true, None, _) => {
                return Err(Error::Invalid(
                    "an image denoiser requires a prefix cache".into(),
                ));
            }
            (true, _, None) => {
                return Err(Error::Interface(
                    "an image denoiser reports no denoiser layout".into(),
                ));
            }
        };

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

        let mut encoders = Vec::new();
        let mut encoder_routes = HashMap::new();
        let mut diffusion = None;
        let mut decoder = None;
        let mut units = None;
        // Every runner's Python half constructs its execution context over
        // the shared work areas of its stream, in this order.
        let mut order = Vec::new();
        for target in &targets {
            let scratch = startup.scratches.get(
                target.stream.as_ref().map_or(0, Stream::raw),
                target.device.cuda_index(),
            )?;
            let setup = ImageSetup {
                loaded,
                component: &target.component,
                path: &target.path,
                device: target.device,
                stream: target.stream.clone(),
                scratch,
            };
            match target.kind {
                CallKind::Media(MediaCall::VisionEncoding | MediaCall::LatentEncoding) => {
                    let info = EncoderInfo::describe(loaded, &target.path, dtype)?;
                    let runner = EncoderRunner::image(&setup, info, &encoder_config)?;
                    order.push((ImageRunner::Encoder(encoders.len()), setup.owner()));
                    encoder_routes.insert((target.component.clone(), target.kind), encoders.len());
                    encoders.push(runner);
                }
                CallKind::Media(MediaCall::Denoising) => {
                    let (
                        Some(layout),
                        Some(capacity),
                        Some(bounds),
                        Some(prefix_cache),
                        Some(latents),
                        Some(pool),
                    ) = (
                        layout,
                        capacity.clone(),
                        bounds.as_ref(),
                        prefix_cache,
                        latents.as_ref(),
                        pool.as_ref(),
                    )
                    else {
                        return Err(Error::Invariant(
                            "an image denoiser lost its storage".into(),
                        ));
                    };
                    let samples = latents.workspace()?.clone();
                    let slot_tables = prefix_cache
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
                    let runner = ImageDiffusion::new(
                        &setup,
                        pool,
                        layout,
                        capacity,
                        DiffusionStorage {
                            samples,
                            slot_timesteps: latents.storage.timesteps().clone(),
                            slot_tables,
                            page_tokens,
                            windowed,
                        },
                    )?;
                    order.push((ImageRunner::Denoiser, setup.owner()));
                    units = Some(prefix_cache.bind_units(target.stream.as_ref())?);
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
                    let runner = DecoderRunner::image(&setup, range, latents, staging)?;
                    order.push((ImageRunner::Decoder, setup.owner()));
                    decoder = Some((target.component.clone(), runner));
                }
                _ => {}
            }
        }

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
                        .filter_map(Device::cuda_index)
                        .filter(|&device| device != ordinal),
                );
                flow_pools = reserve_pools(
                    startup.graphs,
                    &Interpreter::import("uniserve.cuda_graph.pool")?,
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
            let create_pool = Interpreter::import("uniserve.cuda_graph.pool")?;
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
        let device = config.device.cuda_index();
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
            model: facts,
            policy,
            encoders: EncoderRunners::image(encoders, encoder_routes, pools),
            denoisers: DenoiserRunners::image(diffusion, flow, flow_pools, units),
            decoders: DecoderRunners::image(decoder, max_pixel_words),
            latents,
            order,
            outputs,
            host,
        }))
    }
}

/// Device bytes an image model's runners allocate beside the prefix cache and latent pools:
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

/// Start the runners of a video model's components placed on this rank, in
/// placement order, and the host media of its host components.
pub(crate) fn start_video(
    mut setup: VideoSetup<'_>,
    components: &Components,
) -> Result<(
    EncoderRunners,
    DenoiserRunners,
    DecoderRunners,
    Option<MediaRunner>,
)> {
    use crate::config::placement::Capability;

    let mut encoders = EncoderRunners::default();
    let mut denoisers = DenoiserRunners::default();
    let mut decoders = DecoderRunners::default();
    let mut host_components = std::collections::BTreeMap::new();
    for placed in components.placed() {
        if placed.host() {
            host_components.insert(placed.name.clone(), placed.config.clone());
            continue;
        }
        if placed.has(Capability::TextEncoder) {
            encoders.start_text(&mut setup, placed)?;
        }
        if placed.name == setup.model.denoiser {
            denoisers.start_video(&mut setup, placed)?;
        }
        if placed.has(Capability::VideoDecoder) {
            decoders.start_video(&mut setup, placed)?;
        }
        if placed.has(Capability::AudioDecoder) {
            decoders.start_audio(&mut setup, placed)?;
        }
        if placed.has(Capability::VideoEncoder) {
            encoders.start_conditions(&mut setup, placed)?;
        }
    }
    let media = if host_components.is_empty() {
        None
    } else {
        let info = setup.info;
        let capacity = (info.queue_depth as usize * info.max_batch_calls.max(1) as usize).max(1);
        Some(MediaRunner::new(
            setup.loaded,
            host_components,
            setup.config.ffmpeg.clone(),
            setup.model.facts.frame_rate,
            capacity,
        )?)
    };
    Ok((encoders, denoisers, decoders, media))
}

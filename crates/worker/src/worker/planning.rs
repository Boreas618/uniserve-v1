//! Capacity planning of a loaded rank ([`plan`]): the block size, request slots,
//! cache units and latent pages its device budget holds, agreed over the group,
//! with a video rank's slot layout, and the planned capacity reported once for
//! operators.

use std::collections::BTreeMap;
use std::sync::Arc;

use cuda::HandleType;
use ipc::{CallKind, ComponentInfo};

use super::loading::LoadedModel;
use super::{endpoint, expert_exchange, exports_device_products};
use crate::capacity::device::{DeviceBudget, total_bytes};
use crate::capacity::plan::{Plan, Report, TokenFacts, TokenLayout, resolve_block_size};
use crate::capacity::video::VideoCapacity;
use crate::config::placement::Capability;
use crate::config::placement::Components;
use crate::config::{Device, WorkerConfig, WorkerLaunch};
use crate::ffi::DType;
use crate::memory::{CachePlan, GroupShape, LatentPlan, PrefixCache, UnitColumn};
use crate::parallel::world::World;
use crate::runner::causal_lm::start::reserved_bytes;
use crate::runner::denoiser::VideoModel;
use crate::runner::denoiser::image_model::ImageModel;
use crate::runner::diffusion_lm::{self};
use crate::runner::encoder::EncoderInfo;
use crate::runner::start::reserved_bytes as image_reserved_bytes;
use crate::{Error, Result};

/// Block-table width that holds the longest admitted sequence in every
/// group.
pub(super) fn cache_width(shapes: &[GroupShape], max_model_len: usize) -> u32 {
    shapes
        .iter()
        .map(|shape| shape.resident_width(max_model_len as u64))
        .max()
        .unwrap_or(1)
        .try_into()
        .unwrap_or(u32::MAX)
}

/// Report the planned capacity once, for operators.
pub(super) fn log_capacity(config: &WorkerConfig, plan: &Plan) {
    let info = &plan.info;
    // The unit size is the rank's own layout fact (its cache plan); the
    // report carries only the agreed unit count and the groups.
    let prefix_cache = info
        .prefix_cache
        .as_ref()
        .zip(plan.cache.as_ref())
        .map_or_else(
            || "no cache pool".to_owned(),
            |(cache, layout)| {
                format!(
                    "{} cache units of {} bytes in {} groups",
                    cache.num_units,
                    layout.unit_bytes,
                    cache.groups.len()
                )
            },
        );
    tracing::info!(
        "planned capacity: {prefix_cache}, KV block size {} tokens, {} request slots, {} calls and {} tokens per batch, {} latent pages, {} buffer pool bytes",
        config.block_size.unwrap_or(0),
        info.request_slots,
        info.max_batch_calls,
        info.max_batch_tokens,
        info.latent_pages,
        info.buffer_pool_bytes,
    );
}

/// Bytes of the largest vision feature product the image policy admits: the
/// feature rows of its largest image at the vision encoder's output width.
/// Zero for a model without a vision encoder or an understanding transform.
fn vision_feature_bytes(model: &LoadedModel, model_dtype: &str) -> Result<u64> {
    let declarations = &model.declarations;
    let Some(call) = declarations
        .names()
        .flat_map(|name| declarations.calls(name))
        .find(|call| call.is(Capability::PatchEncoder))
    else {
        return Ok(0);
    };
    let dtype = DType::from_name(model_dtype)
        .ok_or_else(|| Error::Unsupported(format!("unsupported model dtype {model_dtype:?}")))?;
    let EncoderInfo::Vision(vision) = EncoderInfo::describe(model.handle(), &call.path, dtype)?
    else {
        return Ok(0);
    };

    // The encoder merges `downsample` patches on each axis into one feature.
    let rows = u64::from(vision.image_patches / (vision.downsample * vision.downsample));
    Ok(rows * u64::from(vision.output_size) * vision.output_dtype.bytes() as u64)
}

/// Resolve the rank's capacity on the loaded model.
#[expect(
    clippy::too_many_arguments,
    reason = "the loaded rank's owners that capacity planning reads"
)]
pub(super) fn plan(
    launch: &WorkerLaunch,
    config: &mut WorkerConfig,
    model: &LoadedModel,
    image: &ImageModel,
    world: &World,
    budget: Option<&DeviceBudget>,
    components: &Components,
    unit_columns: usize,
    video: Option<&VideoModel>,
) -> Result<(Plan, Option<VideoCapacity>)> {
    let device = config.device;
    if let (Device::Cuda(index), Some(budget)) = (device, budget) {
        config.pool_storage_bytes =
            Some(budget.budget(index, config.mem_fraction_static)?.available as usize);
    }

    let declarations = &model.declarations;
    let text = declarations.has(Capability::CausalLm);
    let prefix_cache_dtype = config.prefix_cache_dtype.clone();
    let block_size = resolve_block_size(
        config.block_size,
        |size| {
            if text {
                CachePlan::reads_pages(
                    model.handle(),
                    &model.attention_backend,
                    size,
                    prefix_cache_dtype.as_deref(),
                )
            } else {
                Ok(true)
            }
        },
        &world.host,
    )?;
    config.block_size = Some(block_size);

    let dedicated = config.role == "experts";
    let held: Vec<&str> = launch.components.keys().map(String::as_str).collect();
    // A rank serves every call kind its placed components implement, which
    // for a model rank always includes tensor transfer.
    let supported: Vec<CallKind> = if dedicated {
        Vec::new()
    } else {
        let declared = declarations.supported(&held);
        CallKind::ALL
            .into_iter()
            .filter(|kind| declared.contains(kind))
            .collect()
    };
    let media_components = if dedicated {
        BTreeMap::new()
    } else {
        declarations
            .routes(&held)?
            .into_iter()
            .filter(|(kind, _)| supported.contains(&CallKind::Media(*kind)))
            .collect()
    };
    let capacity = match video {
        Some(video) => {
            let sampling = launch.components.contains_key(&video.denoiser);
            let denoising = components.get(&video.denoiser).is_some();
            let decoding = components
                .placed()
                .iter()
                .any(|placed| placed.has(Capability::VideoDecoder));
            Some(crate::capacity::video::plan(
                video, config, sampling, denoising, decoding,
            )?)
        }
        None => None,
    };
    let infos: Vec<ComponentInfo> = launch
        .components
        .keys()
        .map(|name| ComponentInfo {
            name: name.clone(),
            // Only a video model's components publish products.
            outputs: capacity
                .as_ref()
                .and_then(|capacity| capacity.outputs.get(name).cloned())
                .unwrap_or_default(),
        })
        .collect();
    let fabric_handles = match device {
        Device::Cuda(cuda) => HandleType::probe(cuda)? == HandleType::Fabric,
        Device::Cpu => false,
    };
    // The rank owns its numerics: no other process decides anything from
    // them, so they are stated once, here, rather than in its report.
    tracing::info!(
        dtype = %config.model_dtype,
        attention = %model.attention_backend,
        weights = ?model.weight_formats()?,
        activations = ?model.activation_formats()?,
        "loaded numerics"
    );
    let report = Report {
        endpoint: endpoint(&launch.host),
        device_exports: exports_device_products(launch),
        fabric_handles,
        supported_calls: supported,
        media_components,
        components: infos,
    };
    if let (Some(video), Some(capacity)) = (video, capacity) {
        let plan = video_layout(
            report,
            config,
            launch,
            declarations,
            components,
            video,
            &capacity,
            budget,
            &world.host,
        )?;
        return Ok((plan, Some(capacity)));
    }

    let latent_dtype = DType::from_name(&config.model_dtype).ok_or_else(|| {
        Error::Unsupported(format!("unsupported latent dtype {:?}", config.model_dtype))
    })?;
    // The longest image sequence an image denoiser accepts, in latent rows.
    let image_tokens = image
        .denoiser
        .map_or(0, |layout| layout.max_sequence_tokens as usize);
    let latent = image.denoiser.map(|layout| {
        LatentPlan::image(
            config.max_request_pool_size,
            image_tokens,
            config.max_total_tokens,
            block_size,
            layout.width() as usize,
            latent_dtype,
        )
    });
    let token_facts = TokenFacts {
        image_processor: image.processing.is_some(),
        vision_feature_bytes: vision_feature_bytes(model, &config.model_dtype)?,
        latent_feature_bytes: latent.map_or(0, |plan| {
            image_tokens as u64 * plan.latent_width as u64 * plan.dtype.bytes() as u64
        }),
        latent,
    };
    let mut layout = TokenLayout::new(report, config, launch.ipc.queue_depth, &token_facts)?;
    // Every rank of a worker holding a causal LM component keeps the same
    // cache pool, including ranks that hold only its other components: the
    // worker advertises token work for all of its ranks, and they agree on
    // one pool capacity.
    let holds_text = held.iter().any(|name| {
        declarations
            .calls(name)
            .iter()
            .any(|call| call.is(Capability::CausalLm))
    });
    let cache = if text && !dedicated && holds_text {
        let cache = CachePlan::plan(model.handle(), block_size, prefix_cache_dtype.as_deref())?;
        let total = match device {
            Device::Cuda(index) => total_bytes(index)?,
            Device::Cpu => 0,
        };
        // The cache's fixed storage beside its units: block tables, import
        // scratch and one unit index column per import worker and lane.
        let shapes = cache.group_shapes()?;
        let width = cache_width(&shapes, config.max_model_len);
        let import_capacity = layout.info().max_unresolved_calls as usize;
        let fixed = PrefixCache::fixed_bytes(
            &cache,
            0,
            layout.info().request_slots,
            width,
            import_capacity,
            unit_columns,
        )?;
        layout.reserve(config.device, fixed);
        // The text runners' columns and the slot continuation state.
        layout.reserve(
            config.device,
            reserved_bytes(
                model,
                config,
                layout.info().request_slots as usize,
                &cache,
                image.processing.is_some(),
                unit_columns,
            )?,
        );
        // The image denoiser's columns and the image decoder's staging.
        if image.denoiser.is_some() {
            layout.reserve(
                config.device,
                image_reserved_bytes(
                    config,
                    layout.info().request_slots as usize,
                    image.denoiser.as_ref(),
                ),
            );
        }
        // The diffusion LM runners' buffers, banks and sampler workspaces.
        layout.reserve(
            config.device,
            diffusion_lm::reserved_bytes(
                model,
                components,
                expert_exchange(launch),
                config,
                layout.info().request_slots as usize,
                &cache,
            )?,
        );
        // Every unit column holds one index per unit, charged with each unit:
        // the import workers', the lanes' and the copy lane's of a rank that
        // copies pages through the host KV cache.
        let columns = PrefixCache::import_workers(import_capacity)
            + unit_columns
            + usize::from(launch.storage.host_kv_cache);
        let unit_overhead = columns as u64 * UnitColumn::bytes(1);
        layout.fit_cache(&cache, unit_overhead, config, total, &world.host)?;
        Some(Arc::new(cache))
    } else {
        None
    };
    Ok((layout.finish(config, cache), None))
}

/// The layout of a video rank: request slots sized by every rank's storage
/// for that many resident requests, agreed over the group.
#[expect(
    clippy::too_many_arguments,
    reason = "the placement, model and capacity facts a video rank's layout reads"
)]
fn video_layout(
    report: Report,
    config: &mut WorkerConfig,
    launch: &WorkerLaunch,
    declarations: &crate::config::placement::Declarations,
    components: &Components,
    video: &VideoModel,
    capacity: &VideoCapacity,
    budget: Option<&DeviceBudget>,
    group: &crate::parallel::host::HostGroup,
) -> Result<Plan> {
    let queue_depth = launch.ipc.queue_depth;
    let deployment: Vec<&str> = if config.deployment_components.is_empty() {
        launch.components.keys().map(String::as_str).collect()
    } else {
        config
            .deployment_components
            .iter()
            .map(String::as_str)
            .collect()
    };
    let routes = declarations.routes(&deployment)?;
    let held: std::collections::BTreeSet<String> = components
        .placed()
        .iter()
        .map(|placed| placed.name.clone())
        .collect();
    let local = crate::capacity::video::local_product_bytes(&capacity.outputs, &routes, &held);
    let codec_ranks = launch
        .components
        .get("video_codec")
        .map(|component| component.ranks.len());
    if let (Device::Cuda(index), Some(budget)) = (config.device, budget) {
        let available = budget.budget(index, config.mem_fraction_static)?.available;
        // Slot fitting searches from one slot: a media request pipelines in
        // a single slot's three queue positions.
        let maximum = config.max_request_pool_size;
        let minimum = 1;
        let facts = capacity.facts(video, maximum, local, codec_ranks)?;
        let requirements = crate::capacity::plan::slot_requirements(
            config,
            queue_depth,
            &facts,
            exports_device_products(launch),
            minimum,
            maximum,
        )?;
        let slots = crate::capacity::plan::request_slots(&requirements, minimum, available, group)?;
        config.max_request_pool_size = slots;
        config.max_batch_calls = config.max_batch_calls.min(slots);
        config.max_batch_tokens = config.max_batch_tokens.min(slots);
    }
    let facts = capacity.facts(video, config.max_request_pool_size, local, codec_ranks)?;
    crate::capacity::plan::request_layout(
        report,
        config,
        queue_depth,
        &facts,
        components.holds_host(),
    )
}

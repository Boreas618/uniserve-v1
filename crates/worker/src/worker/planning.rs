//! Capacity planning of a loaded rank: the block size and the cache's layout
//! ([`block_size`], [`cache_plan`]), which the executor's token computations
//! are sized by, then ([`plan`]) the request slots, cache units and latent
//! pages its device budget holds, agreed over the group, with a video rank's
//! slot layout, and the planned capacity reported once for operators.

use std::collections::BTreeMap;
use std::sync::Arc;

use cuda::HandleType;
use ipc::{CallKind, ComponentInfo};

use super::computations::PageManifest;
use super::image::ImageManifest;
use super::loading::LoadedModel;
use super::{endpoint, exports_device_products};
use crate::capacity::device::{DeviceBudget, total_bytes};
use crate::capacity::plan::{Plan, Report, TokenFacts, TokenLayout, resolve_block_size};
use crate::capacity::video::{VideoCapacity, VideoManifest};
use crate::config::placement::Capability;
use crate::config::placement::Components;
use crate::config::{Device, WorkerConfig, WorkerLaunch};
use crate::memory::{CachePlan, GroupShape, LatentPlan, PrefixCache, UnitColumn};
use crate::parallel::world::World;
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

/// Unit columns the computations over the prefix cache bind beside the import
/// workers': one, on the device's primary stream.
const COMPUTATION_COLUMNS: usize = 1;

/// Initial writable cache storage at the target layout. Explicit token
/// capacity is reserved in full on a new GPU; automatic capacity must at
/// least hold the sentinel and the pages needed to advance a request.
pub(super) fn cache_budget(
    launch: &WorkerLaunch,
    config: &WorkerConfig,
    plan: &Plan,
    budget: &mut storage::manifest::weights::RankBudget,
) -> Result<()> {
    let Some(cache) = plan.cache.as_ref() else {
        return Ok(());
    };
    let units = 1 + cache.progress_units();
    let units = match config.device {
        Device::Cuda(device) => crate::memory::prefix_cache::planes::PlaneLayout::whole_granules(
            units,
            cache.plane_bytes,
            HandleType::Descriptor.granularity(device)? as u64,
        ),
        Device::Cpu => units,
    };
    let columns = PrefixCache::import_workers(plan.info.max_unresolved_calls as usize)
        + COMPUTATION_COLUMNS
        + usize::from(launch.storage.host_kv_cache);
    let unit_bytes = cache.unit_bytes + columns as u64 * UnitColumn::bytes(1);
    budget.cache = u64::from(units) * unit_bytes;
    budget.cache_capacity = if config.max_total_tokens.is_some() {
        plan.info
            .prefix_cache
            .as_ref()
            .map_or(0, |cache| u64::from(cache.num_units) * unit_bytes)
    } else {
        budget.cache
    };
    budget.pages = cache
        .groups
        .iter()
        .filter(|group| !group.draft)
        .map(|group| storage::manifest::weights::CachePage {
            tokens: group.block_size,
            bytes: u64::from(group.units_per_page) * unit_bytes,
            window: group.window,
            state: group.state,
        })
        .collect();
    Ok(())
}

/// Resolve the rank's KV block size into `config`: the largest of at most
/// 64 tokens every resident cache layer's attention reads, agreed over the
/// worker group. Collective, at the same point of every rank's startup.
pub(super) fn block_size(
    config: &mut WorkerConfig,
    model: &LoadedModel,
    world: &World,
) -> Result<()> {
    let text = model.declarations.has(Capability::CausalLm);
    let prefix_cache_dtype = config.prefix_cache_dtype.clone();
    let draft = config.draft_tokens > 0;
    let block_size = resolve_block_size(
        config.block_size,
        |size| {
            if text {
                CachePlan::reads_pages(
                    model.handle(),
                    &model.attention_backend,
                    size,
                    prefix_cache_dtype.as_deref(),
                    draft,
                )
            } else {
                Ok(true)
            }
        },
        &world.host,
    )?;
    config.block_size = Some(block_size);
    Ok(())
}

/// The cache layout of a rank of a worker holding a causal language model
/// component, at the resolved block size ([`block_size`]); `None` for any
/// other rank.
///
/// Every rank of such a worker keeps the same cache pool, including ranks
/// that hold only its other components: the worker advertises token work
/// for all of its ranks, and they agree on one pool capacity.
pub(super) fn cache_plan(
    launch: &WorkerLaunch,
    config: &WorkerConfig,
    model: &LoadedModel,
) -> Result<Option<CachePlan>> {
    let declarations = &model.declarations;
    let holds_text = launch.components.keys().any(|name| {
        declarations
            .calls(name)
            .iter()
            .any(|call| call.is(Capability::CausalLm))
    });
    if config.role == "experts" || !declarations.has(Capability::CausalLm) || !holds_text {
        return Ok(None);
    }
    let block_size = config.block_size.ok_or(Error::State(
        "the cache is planned at the resolved block size",
    ))?;
    Ok(Some(CachePlan::plan(
        model.handle(),
        block_size,
        config.prefix_cache_dtype.as_deref(),
        config.draft_tokens > 0,
    )?))
}

/// Resolve the rank's capacity on the loaded model, at the block size and with
/// the `cache` layout resolved before: what the executor's manifest states its
/// computations over the prefix cache allocate (`pages`), and a video model's
/// `video` manifest.
#[expect(
    clippy::too_many_arguments,
    reason = "the loaded rank's owners that capacity planning reads"
)]
pub(super) fn plan(
    launch: &WorkerLaunch,
    config: &mut WorkerConfig,
    model: &LoadedModel,
    image: Option<&ImageManifest>,
    image_processor: bool,
    world: &World,
    budget: Option<&DeviceBudget>,
    components: &Components,
    cache: Option<CachePlan>,
    pages: Option<&PageManifest>,
    video: Option<(&str, VideoManifest)>,
) -> Result<(Plan, Option<VideoCapacity>)> {
    let device = config.device;
    if let (Device::Cuda(index), Some(budget)) = (device, budget) {
        // The grant is what the GPU's ledger leaves beside every world of
        // the deployment; a successor's full grant is what it leaves once
        // its predecessor is gone.
        let budget = budget.budget(index, config.mem_fraction_static)?;
        config.pool_storage_bytes = Some(budget.available as usize);
        config.pool_storage_full_bytes = Some(budget.full as usize);
    }
    let block_size = config
        .block_size
        .ok_or(Error::State("planning follows the block size's resolution"))?;

    let declarations = &model.declarations;
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
        Some((denoiser, manifest)) => {
            let sampling = launch.components.contains_key(denoiser);
            Some(VideoCapacity::new(manifest, sampling)?)
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
                .and_then(|capacity| capacity.outputs().get(name).cloned())
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
    if let Some(capacity) = capacity {
        let plan = video_layout(
            report,
            config,
            launch,
            declarations,
            components,
            &capacity,
            budget,
            &world.host,
        )?;
        return Ok((plan, Some(capacity)));
    }

    // An image denoiser's latent pool holds its longest image sequence, in
    // latent rows, at the executor's latent dtype.
    let latent = image
        .and_then(|image| image.latent.as_ref())
        .map(|latent| {
            LatentPlan::image(
                config.max_request_pool_size,
                latent.rows,
                config.max_total_tokens,
                block_size,
                latent.width,
                latent.dtype()?,
            )
        })
        .transpose()?;
    let token_facts = TokenFacts {
        image_processor,
        vision_feature_bytes: image.map_or(0, |image| image.vision_feature_bytes),
        latent_feature_bytes: image.map_or(0, |image| image.latent_feature_bytes),
        latent,
    };
    let mut layout = TokenLayout::new(report, config, launch.ipc.queue_depth, &token_facts)?;
    let cache = if let Some(cache) = cache {
        let total = match device {
            Device::Cuda(index) => total_bytes(index)?,
            Device::Cpu => 0,
        };
        // The cache's fixed storage beside its units: block tables, import
        // scratch and one unit index column per import worker and per stream
        // the computations over the cache bind.
        let shapes = cache.group_shapes()?;
        let width = cache_width(&shapes, config.max_model_len);
        let import_capacity = layout.info().max_unresolved_calls as usize;
        let fixed = PrefixCache::fixed_bytes(
            &cache,
            0,
            layout.info().request_slots,
            width,
            import_capacity,
            COMPUTATION_COLUMNS,
        )?;
        layout.reserve(config.device, fixed);
        // The buffers of the computations over the prefix cache and their
        // request slots' state.
        if let Some(pages) = pages {
            layout.reserve(
                config.device,
                pages.reserved_bytes(layout.info().request_slots as usize),
            );
        }
        // The image computations' buffers and their request slots' state.
        if let Some(image) = image {
            layout.reserve(
                config.device,
                image.reserved_bytes(layout.info().request_slots as usize),
            );
        }
        // Every unit column holds one index per unit, charged with each unit:
        // the import workers', the primary stream's and the copy stream's of a
        // rank that copies pages through the host KV cache.
        let columns = PrefixCache::import_workers(import_capacity)
            + COMPUTATION_COLUMNS
            + usize::from(launch.storage.host_kv_cache);
        let unit_overhead = columns as u64 * UnitColumn::bytes(1);
        let granularity = match config.device {
            Device::Cuda(device) => Some(HandleType::Descriptor.granularity(device)? as u64),
            Device::Cpu => None,
        };
        layout.fit_cache(
            &cache,
            unit_overhead,
            config,
            total,
            granularity,
            &world.host,
        )?;
        Some(Arc::new(cache))
    } else {
        None
    };
    Ok((layout.finish(cache), None))
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
    let local = crate::capacity::video::local_product_bytes(capacity.outputs(), &routes, &held);
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
        let facts = capacity.facts(maximum, local, codec_ranks)?;
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
    let facts = capacity.facts(config.max_request_pool_size, local, codec_ranks)?;
    crate::capacity::plan::request_layout(
        report,
        config,
        queue_depth,
        &facts,
        components.holds_host(),
    )
}

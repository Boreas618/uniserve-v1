//! Startup capacity planning: page size, request slots, cache units, latent
//! pages and the [`WorkerInfo`] the rank reports.
//!
//! Planning runs after the model is loaded, so measured device storage
//! includes the weights. Every quantity the engine schedules against must be
//! identical on the ranks of a group, so the page size, the request-slot fit
//! and the cache unit count are agreed over the group's [`HostGroup`] with a
//! minimum reduction.
//!
//! Token workers (text, diffusion LM, image) place a cache pool beside fixed storage:
//! the buffer pool, the tensor store's products, the latent pool and the
//! storage each capability runtime reserves (input buffers, block tables,
//! decode state, importer scratch). Automatic cache sizing gives the pool the
//! grant left after those and a graph-capture reservation. Media workers
//! (video) size request slots instead, because every resident request keeps
//! its whole media state.

use std::collections::BTreeMap;

use std::sync::Arc;

use uniserve_ipc::{
    CallKind, ComponentInfo, MediaCall, OutputInfo, VideoDenoiserInfo, WorkerEndpoint, WorkerInfo,
};

use super::{
    ArenaCapacity, CacheCapacity, CacheDemand, RequestBounds, TokenBounds, call_window,
    request_tensor_window,
};
use crate::config::{Device, WorkerConfig};
use crate::cuda_graph::graph_storage_budget_bytes;
use crate::parallel::host::HostGroup;
use crate::storage::{CachePlan, LatentPlan};
use crate::{Error, Result};

/// Bytes one product slot of a component's outputs retains: every output's
/// largest extent, aligned to 256 bytes.
pub(crate) fn product_bytes(outputs: &[OutputInfo]) -> u64 {
    outputs
        .iter()
        .map(|output| {
            (output.shape_bound.max_elements() * output.dtype.element_bytes()).div_ceil(256) * 256
        })
        .sum()
}

/// The largest page size of at most 64 tokens that every resident layer's
/// attention reads, agreed over `group` as the minimum.
///
/// A preset size is used unchanged and needs no agreement: every rank was
/// launched with it. `readable` plans the pool for one candidate size.
pub(crate) fn page_size(
    preset: Option<usize>,
    mut readable: impl FnMut(usize) -> Result<bool>,
    group: &HostGroup,
) -> Result<usize> {
    if let Some(size) = preset {
        return Ok(size);
    }
    let mut size = 64;
    while !readable(size)? {
        size /= 2;
        if size == 0 {
            return Err(Error::Unsupported(
                "no KV page size of at most 64 tokens is read by the attention of every resident cache layer".into(),
            ));
        }
    }
    let agreed = group.all_reduce_min(&[size as i64])?;
    Ok(agreed[0] as usize)
}

/// The largest request-slot count every rank fits.
///
/// `requirements[i]` is this rank's storage for `minimum + i` slots. Every
/// rank tests the same candidates, because a smaller slot count can lengthen
/// the output horizon, so reducing local maxima would miss a common fit.
pub(crate) fn request_slots(
    requirements: &[u64],
    minimum: usize,
    available: u64,
    group: &HostGroup,
) -> Result<usize> {
    if minimum == 0 || requirements.is_empty() {
        return Err(Error::Invalid(
            "request tensor capacity requires valid slot bounds".into(),
        ));
    }
    let fits: Vec<i64> = requirements
        .iter()
        .map(|&bytes| i64::from(bytes <= available))
        .collect();
    let fits = group.all_reduce_min(&fits)?;
    fits.iter()
        .rposition(|&fits| fits != 0)
        .map(|offset| minimum + offset)
        .ok_or_else(|| {
            Error::Unsupported(format!(
                "insufficient device storage for a common request tensor slot count: candidate range {minimum}..{}, local requirements {requirements:?}, {available} bytes available",
                minimum + requirements.len() - 1,
            ))
        })
}

impl LatentPlan {
    /// The pool of an image flow: pages of `block_size` units holding the
    /// longest image (or the configured token budget, when larger).
    pub(crate) fn image(
        slots: usize,
        max_tokens: usize,
        max_total_tokens: Option<usize>,
        block_size: usize,
        width: usize,
        dtype: crate::ffi::DType,
    ) -> Self {
        let units = if max_tokens == 0 {
            0
        } else {
            max_tokens.max(max_total_tokens.unwrap_or(max_tokens))
        };
        Self {
            request_pool_size: slots,
            num_pages: units.div_ceil(block_size) + 1,
            page_units: block_size,
            latent_width: width,
            dtype,
            with_workspace: true,
        }
    }
}

/// Model-independent values of the report.
pub(crate) struct Report {
    pub model_name: String,
    pub endpoint: WorkerEndpoint,
    pub device: String,
    pub transfer_backends: Vec<String>,
    /// The rank publishes device products over CUDA VMM, which keeps an
    /// export pool on its primary device.
    pub device_exports: bool,
    pub fabric_handles: bool,
    pub attention_backend: String,
    pub supported_calls: Vec<CallKind>,
    pub media_components: BTreeMap<MediaCall, String>,
    pub components: Vec<ComponentInfo>,
    pub weight_formats: Vec<String>,
    pub activation_formats: Vec<String>,
}

/// Media-timeline facts of a video model.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MediaFacts {
    pub num_steps: u32,
    pub video_denoiser: Option<VideoDenoiserInfo>,
    /// Device bytes of one request's state buffers.
    pub state_bytes: u64,
    /// Product bytes this rank backs per request: those of the components it
    /// executes and of the components it reads.
    pub local_product_bytes: u64,
    /// Latent pool of sampled trajectories, when the worker holds the
    /// denoiser. Every rank of the worker reports its pages, because the head
    /// schedules a request's pages by the worker's report.
    pub latent: Option<LatentPlan>,
    /// Whether this rank stores the latent pool: it holds the denoiser.
    pub stores_latent: bool,
    /// Artifact regions imported from other codec ranks.
    pub artifact_import_regions: usize,
}

impl MediaFacts {
    /// The latent pool this rank stores.
    fn stored_latent(&self) -> Option<LatentPlan> {
        self.latent.filter(|_| self.stores_latent)
    }
}

/// Inputs of the token layout taken from the loaded model.
pub(crate) struct TokenFacts {
    /// The model has an image processor (it encodes request images).
    pub image_processor: bool,
    /// Bytes of one encoded image's vision features.
    pub vision_feature_bytes: u64,
    /// Bytes of one image's latent features (image flow).
    pub latent_feature_bytes: u64,
    /// The latent pool of an image flow.
    pub latent: Option<LatentPlan>,
}

/// The resolved capacity of a rank.
pub(crate) struct Plan {
    pub info: WorkerInfo,
    pub arena: ArenaCapacity,
    /// Fixed bytes by device, beside the cache pool.
    pub fixed_device_bytes: BTreeMap<Device, u64>,
    pub latent: Option<LatentPlan>,
    /// The cache layout, with the agreed unit count in `info.prefix_cache`.
    pub cache: Option<Arc<CachePlan>>,
}

impl Plan {
    /// Every eligible lane must be able to execute an admitted batch, so
    /// lane limits bound the advertised batch; storage still covers warmup
    /// and graph capture at the configured dimensions.
    fn clamp_lanes(&mut self, config: &WorkerConfig) {
        for lane in &config.lanes {
            if let Some(calls) = lane.max_batch_calls {
                self.info.max_batch_calls = self.info.max_batch_calls.min(saturate(calls));
            }
            if let Some(tokens) = lane.max_batch_tokens {
                self.info.max_batch_tokens = self.info.max_batch_tokens.min(saturate(tokens));
            }
        }
    }
}

fn saturate(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

fn narrow(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::Invalid(format!("{name} exceeds the protocol range")))
}

fn base_info(report: Report, config: &WorkerConfig, queue_depth: usize) -> Result<WorkerInfo> {
    Ok(WorkerInfo {
        model_name: report.model_name,
        media_components: report.media_components,
        num_inference_steps: 0,
        video_denoiser: None,
        endpoint: report.endpoint,
        device: report.device,
        transfer_backends: report.transfer_backends,
        fabric_handles: report.fabric_handles,
        world_size: narrow(config.world_size, "world size")?,
        model_dtype: config.model_dtype.clone(),
        attention_backend: report.attention_backend,
        weight_formats: report.weight_formats,
        activation_formats: report.activation_formats,
        components: report.components,
        supported_calls: report.supported_calls,
        queue_depth: narrow(queue_depth, "queue depth")?,
        max_batch_calls: narrow(config.max_batch_calls, "max batch calls")?,
        max_batch_tokens: narrow(config.max_batch_tokens, "max batch tokens")?,
        request_slots: narrow(config.max_request_pool_size, "request slots")?,
        prefix_cache: None,
        latent_page_units: 0,
        latent_pages: 0,
        buffer_pool_bytes: 0,
        encoder_cache_entries: 0,
        encoder_entry_bytes: 0,
        max_unresolved_calls: 0,
        host_lane_capacity: 1,
        transfer_tickets: 1,
        image_warmups: Vec::new(),
        diffusion_lm_step_rows: 0,
        image_unconditional_token_ids: None,
    })
}

/// A token worker's layout before the cache pool is fitted.
pub(crate) struct TokenLayout {
    info: WorkerInfo,
    arena: ArenaCapacity,
    fixed: BTreeMap<Device, u64>,
    feature_bytes: u64,
    latent: Option<LatentPlan>,
}

impl TokenLayout {
    /// Size the buffer pool, product store and latent pool.
    pub(crate) fn new(
        report: Report,
        config: &WorkerConfig,
        queue_depth: usize,
        facts: &TokenFacts,
    ) -> Result<Self> {
        let device_exports = report.device_exports;
        let mut info = base_info(report, config, queue_depth)?;
        let latent_bytes = facts
            .latent
            .map(|plan| plan.bytes())
            .transpose()?
            .unwrap_or(0);
        let feature_bytes = facts.latent_feature_bytes.max(facts.vision_feature_bytes);
        info.encoder_cache_entries = if facts.image_processor {
            narrow(config.encoder_cache_entries, "encoder cache entries")?
        } else {
            0
        };
        info.encoder_entry_bytes = if info.encoder_cache_entries == 0 {
            0
        } else {
            feature_bytes
        };
        info.buffer_pool_bytes = (u64::from(info.encoder_cache_entries) + 1) * feature_bytes;
        info.max_unresolved_calls = narrow(
            call_window(queue_depth, config.max_batch_calls)?,
            "unresolved calls",
        )?;
        if let Some(plan) = &facts.latent {
            info.latent_page_units = narrow(plan.page_units, "latent page units")?;
            info.latent_pages = narrow(plan.num_pages, "latent pages")?;
        }

        // Products, their buffer pool and the latent pool live on the primary
        // device, beside every binding; a generation tower holds only the
        // module weights placed on it.
        let arena = ArenaCapacity::for_tokens(&TokenBounds {
            depth: queue_depth,
            calls: config.max_batch_calls,
            requests: config.max_request_pool_size,
            transfer_bytes_per_read: feature_bytes,
            device_exports,
        })?;
        let mut fixed = BTreeMap::new();
        fixed.insert(
            config.device,
            info.buffer_pool_bytes + arena.deferred_device_bytes() + latent_bytes,
        );
        info.host_lane_capacity = narrow(arena.host_lane_inflight, "host lane capacity")?;
        info.transfer_tickets = narrow(arena.transfer_tickets, "transfer tickets")?;
        Ok(Self {
            info,
            arena,
            fixed,
            feature_bytes,
            latent: facts.latent,
        })
    }

    pub(crate) fn info(&self) -> &WorkerInfo {
        &self.info
    }

    /// Record fixed storage a capability runtime allocates on `device`.
    pub(crate) fn reserve(&mut self, device: Device, bytes: u64) {
        *self.fixed.entry(device).or_default() += bytes;
    }

    /// Fit the cache pool beside the fixed storage and agree its unit count.
    ///
    /// The pool takes the grant left after a graph-capture reservation (10%
    /// of the device) and the fixed storage, in whole units. Graph padding
    /// takes no unit: padding rows read the sentinel unit and write no K/V.
    /// A configured token capacity sizes the pool instead. `total_bytes` is
    /// the device's total memory, zero on CPU. `unit_overhead` is device
    /// storage other owners keep per unit (unit index columns), charged with
    /// each unit.
    pub(crate) fn fit_cache(
        &mut self,
        cache: &CachePlan,
        unit_overhead: u64,
        config: &WorkerConfig,
        total_bytes: u64,
        group: &HostGroup,
    ) -> Result<CacheCapacity> {
        let unit_bytes = cache.unit_bytes + unit_overhead;
        let pages: Vec<(u32, u32)> = cache
            .groups
            .iter()
            .map(|group| (group.page_tokens, group.units_per_page))
            .collect();
        let reserved_units = graph_storage_budget_bytes(total_bytes as i64).div_ceil(unit_bytes)
            + self
                .fixed
                .get(&config.device)
                .copied()
                .unwrap_or(0)
                .div_ceil(unit_bytes);

        let cuda = config.device.is_cuda();
        let tokens = config.max_total_tokens.map(|tokens| tokens as u64);
        let available = config.pool_storage_bytes.map(|bytes| bytes as u64);
        if cuda && tokens.is_none() && available.is_none() {
            return Err(Error::Invalid(
                "automatic CUDA KV sizing requires a host storage grant".into(),
            ));
        }
        let mut capacity = CacheCapacity::derive(&CacheDemand {
            pages: &pages,
            tokens,
            unit_bytes,
            available_bytes: available,
            reserved_units,
        })?;
        // Every member allocates the same logical unit ids, bounded by the
        // smallest grant.
        capacity.num_units = group.all_reduce_min(&[capacity.num_units as i64])?[0] as u64;
        let num_units = u32::try_from(capacity.num_units)
            .map_err(|_| Error::Invalid("KV pool exceeds the unit index range".into()))?;
        self.info.prefix_cache = Some(cache.info(num_units));
        // The arena's slot counts and storage are independent of cache
        // capacity; only its transfer byte budget follows the fitted pool,
        // since one read may carry a whole pool's export. The budget is
        // accounting: cache exports name the pages where they lie and
        // occupy no storage of their own.
        self.arena.transfer_bytes = (capacity.num_units * cache.unit_bytes)
            .max(self.feature_bytes)
            .max(1)
            * self.arena.transfer_tickets as u64;
        Ok(capacity)
    }

    pub(crate) fn finish(self, config: &WorkerConfig, cache: Option<Arc<CachePlan>>) -> Plan {
        let mut plan = Plan {
            info: self.info,
            arena: self.arena,
            fixed_device_bytes: self.fixed,
            latent: self.latent,
            cache,
        };
        plan.clamp_lanes(config);
        plan
    }
}

/// The layout of a media worker, sized by its request slots.
pub(crate) fn request_layout(
    report: Report,
    config: &WorkerConfig,
    queue_depth: usize,
    media: &MediaFacts,
    host_components: bool,
) -> Result<Plan> {
    let device_exports = report.device_exports;
    let mut info = base_info(report, config, queue_depth)?;
    let slots = info.request_slots as usize;
    info.max_unresolved_calls = narrow(
        request_tensor_window(queue_depth, slots)?,
        "unresolved calls",
    )?;
    info.max_batch_calls = info.max_batch_calls.min(info.request_slots);
    info.max_batch_tokens = info.max_batch_calls;
    // The pool this rank backs, which the engine places this rank's buffers
    // in: the products of the components it executes and reads.
    info.buffer_pool_bytes = slots as u64 * media.local_product_bytes;
    info.num_inference_steps = media.num_steps;
    info.video_denoiser = media.video_denoiser.clone();
    if let Some(plan) = &media.latent {
        info.latent_page_units = narrow(plan.page_units, "latent page units")?;
        info.latent_pages = narrow(plan.num_pages, "latent pages")?;
    }
    let arena = ArenaCapacity::for_requests(&RequestBounds {
        depth: queue_depth,
        calls: info.max_batch_calls as usize,
        requests: slots,
        ranks: config.world_size,
        product_bytes_per_request: media.local_product_bytes,
        concurrent_imports: media.artifact_import_regions,
        device_exports,
    })?;
    info.host_lane_capacity = if host_components {
        1
    } else {
        narrow(arena.host_lane_inflight, "host lane capacity")?
    };
    info.transfer_tickets = narrow(arena.transfer_tickets, "transfer tickets")?;
    let mut plan = Plan {
        info,
        arena,
        fixed_device_bytes: BTreeMap::new(),
        latent: media.stored_latent(),
        cache: None,
    };
    plan.clamp_lanes(config);
    Ok(plan)
}

/// This rank's storage for every candidate slot count from `minimum` to
/// `maximum`: state, retained products, the product store with its export
/// pool (when `device_exports`) and the latent pool of each count.
pub(crate) fn slot_requirements(
    config: &WorkerConfig,
    queue_depth: usize,
    media: &MediaFacts,
    device_exports: bool,
    minimum: usize,
    maximum: usize,
) -> Result<Vec<u64>> {
    let mut requirements = Vec::new();
    for count in minimum..=maximum {
        let arena = ArenaCapacity::for_requests(&RequestBounds {
            depth: queue_depth,
            calls: config.max_batch_calls.min(count),
            requests: count,
            ranks: config.world_size,
            product_bytes_per_request: media.local_product_bytes,
            concurrent_imports: 0,
            device_exports,
        })?;
        let latent_bytes = match media.stored_latent() {
            Some(plan) => {
                let pages = if plan.with_workspace {
                    plan.num_pages
                } else {
                    count * ((plan.num_pages - 1) / plan.request_pool_size) + 1
                };
                LatentPlan {
                    request_pool_size: count,
                    num_pages: pages,
                    ..plan
                }
                .bytes()?
            }
            None => 0,
        };
        requirements.push(
            count as u64 * (media.state_bytes + media.local_product_bytes)
                + arena.deferred_device_bytes()
                + latent_bytes,
        );
    }
    Ok(requirements)
}

#[cfg(test)]
pub(crate) mod tests {
    use uniserve_cuda::DeviceId;
    use uniserve_ipc::{DType, DimBound, ShapeBound};

    use super::*;
    use crate::capacity::DEFAULT_NUM_UNITS;

    fn group() -> HostGroup {
        HostGroup::single(0)
    }

    /// The configuration of a one-rank CPU worker launched with the head's
    /// defaults; each test states the values it plans with.
    pub(crate) fn cpu_config() -> WorkerConfig {
        WorkerConfig {
            device: Device::Cpu,
            rank: 0,
            world_size: 1,
            role: "model".into(),
            expert_exchange: "alltoall".into(),
            expert_microbatches: 1,
            block_size: None,
            max_total_tokens: None,
            attention_backend: "auto".into(),
            max_batch_calls: 1024,
            max_batch_tokens: 8192,
            max_model_len: 16384,
            max_video_seconds: 15.0,
            max_condition_rows: 0,
            ffmpeg: "ffmpeg".into(),
            min_video_seconds: None,
            video_text_capacities: Vec::new(),
            video_canvases: Vec::new(),
            deployment_components: Vec::new(),
            canvas_sampling: None,
            max_request_pool_size: 128,
            encoder_cache_entries: 256,
            generation_device: None,
            pool_storage_bytes: None,
            model_dtype: "bfloat16".into(),
            prefix_cache_dtype: None,
            mem_fraction_static: 0.7,
            lanes: Vec::new(),
            graph_policy: "auto".into(),
            decode_graph_batch_sizes: crate::config::default_decode_sizes(),
            prefill_cuda_graph: true,
            prefill_outputs: true,
            prefill_graph_token_sizes: crate::config::default_prefill_sizes(),
            flow_cuda_graph: true,
            flow_graph_batch_sizes: crate::config::DEFAULT_FLOW_GRAPH_BATCH_SIZES.to_vec(),
            flow_graph_shapes: crate::config::DEFAULT_FLOW_GRAPH_SHAPES.to_vec(),
        }
    }

    fn report() -> Report {
        Report {
            model_name: "model".into(),
            endpoint: WorkerEndpoint {
                worker_id: "worker".into(),
                rank: 0,
                node: "node".into(),
                address_space: "1:a".into(),
                incarnation: "b".into(),
            },
            device: "cuda:0".into(),
            transfer_backends: vec!["local".into()],
            device_exports: false,
            fabric_handles: false,
            attention_backend: "auto".into(),
            supported_calls: vec![CallKind::Forward(uniserve_ipc::ForwardMode::Prefill)],
            media_components: BTreeMap::new(),
            components: Vec::new(),
            weight_formats: vec!["dense".into()],
            activation_formats: vec!["dense".into()],
        }
    }

    /// A two-layer, one-column bfloat16 pool with 64-token pages.
    fn cache() -> CachePlan {
        let plane_bytes = 64 * 4 * 128 * 2;
        CachePlan {
            columns: 1,
            plane_bytes,
            quantized: false,
            // Key and value planes, each with one initialization flag.
            unit_bytes: 2 * (plane_bytes + 1),
            dtype: uniserve_core::PrefixCacheDtype::BFloat16,
            groups: vec![crate::storage::CacheGroupPlan {
                window: None,
                page_tokens: 64,
                units_per_page: 2,
                layers: vec!["layers.0".into(), "layers.1".into()],
                layer_ids: vec![0, 1],
                axis: crate::storage::GroupAxis {
                    offset: 0,
                    total: 2,
                },
                num_kv_heads: 4,
                total_kv_heads: 8,
                kv_head_offset: 4,
                head_dim: 128,
                dtype: crate::ffi::DType::BF16,
                compute_dtype: crate::ffi::DType::BF16,
            }],
            handle: crate::ffi::Handle::none(),
        }
    }

    #[test]
    fn products_align_every_output_to_256_bytes() {
        let output = |elements: u32| OutputInfo {
            name: "features".into(),
            dtype: DType::BF16,
            shape_bound: ShapeBound {
                dims: vec![DimBound::Device { max: elements }, DimBound::Static(1)],
            },
            raster_axes: None,
        };
        assert_eq!(product_bytes(&[output(100), output(128)]), 256 + 256);
        assert_eq!(product_bytes(&[output(129)]), 512);
    }

    #[test]
    fn the_page_size_halves_until_every_layer_reads_it() {
        let mut probed = Vec::new();
        let size = page_size(
            None,
            |size| {
                probed.push(size);
                Ok(size <= 16)
            },
            &group(),
        )
        .unwrap();
        assert_eq!(size, 16);
        assert_eq!(probed, vec![64, 32, 16]);
        assert_eq!(page_size(Some(8), |_| Ok(false), &group()).unwrap(), 8);
        assert!(page_size(None, |_| Ok(false), &group()).is_err());
    }

    #[test]
    fn request_slots_take_the_largest_common_fit() {
        let requirements = [100, 200, 300, 400];
        assert_eq!(request_slots(&requirements, 2, 250, &group()).unwrap(), 3);
        assert!(request_slots(&requirements, 2, 50, &group()).is_err());
    }

    #[test]
    fn the_cache_pool_takes_the_grant_left_after_reservations() {
        let gib = 1u64 << 30;
        let config = WorkerConfig {
            device: Device::Cuda(DeviceId(0)),
            max_batch_calls: 8,
            max_batch_tokens: 2048,
            max_request_pool_size: 8,
            pool_storage_bytes: Some((20 * gib) as usize),
            ..cpu_config()
        };
        let facts = TokenFacts {
            image_processor: false,
            vision_feature_bytes: 0,
            latent_feature_bytes: 0,
            latent: None,
        };
        let mut layout = TokenLayout::new(report(), &config, 2, &facts).unwrap();
        let fixed_before = layout.fixed[&Device::Cuda(DeviceId(0))];
        layout.reserve(Device::Cuda(DeviceId(0)), gib);
        let cache = cache();
        let capacity = layout
            .fit_cache(&cache, 0, &config, 100 * gib, &group())
            .unwrap();

        let unit = cache.unit_bytes;
        let reserved = (10 * gib).div_ceil(unit) + (fixed_before + gib).div_ceil(unit);
        assert_eq!(capacity.num_units, 20 * gib / unit - reserved);
        let plan = layout.finish(&config, Some(Arc::new(cache)));
        let info = plan.info.prefix_cache.as_ref().unwrap();
        assert_eq!(u64::from(info.num_units), capacity.num_units);
        assert_eq!(info.groups[0].kv_head_offset, 4);
        assert_eq!(
            plan.arena.transfer_bytes,
            capacity.num_units * unit * plan.arena.transfer_tickets as u64
        );
    }

    #[test]
    fn cpu_workers_take_the_default_pool_and_lane_limits_clamp_the_batch() {
        let config = WorkerConfig {
            device: Device::Cpu,
            max_batch_calls: 8,
            max_batch_tokens: 64,
            max_request_pool_size: 8,
            lanes: vec![uniserve_ipc::LaneConfig {
                lane_id: "decode".into(),
                sm_budget: 1,
                call_kinds: vec![CallKind::Forward(uniserve_ipc::ForwardMode::Decode)],
                max_batch_calls: Some(4),
                max_batch_tokens: None,
                max_inflight: None,
            }],
            ..cpu_config()
        };
        let facts = TokenFacts {
            image_processor: true,
            vision_feature_bytes: 1024,
            latent_feature_bytes: 512,
            latent: Some(LatentPlan::image(
                8,
                1000,
                None,
                64,
                12,
                crate::ffi::DType::BF16,
            )),
        };
        let mut layout = TokenLayout::new(report(), &config, 2, &facts).unwrap();
        let capacity = layout.fit_cache(&cache(), 0, &config, 0, &group()).unwrap();
        assert_eq!(capacity.num_units, DEFAULT_NUM_UNITS);
        let plan = layout.finish(&config, Some(Arc::new(cache())));
        assert_eq!(plan.info.max_batch_calls, 4);
        assert_eq!(plan.info.encoder_entry_bytes, 1024);
        // One feature per encoder cache entry and one in flight.
        assert_eq!(plan.info.buffer_pool_bytes, 257 * 1024);
        // 1000 tokens in 64-unit pages, plus the sentinel page.
        assert_eq!(plan.info.latent_pages, 17);
        assert_eq!(plan.info.latent_page_units, 64);
    }

    #[test]
    fn slot_requirements_grow_with_the_slot_count() {
        let config = WorkerConfig {
            max_batch_calls: 4,
            ..cpu_config()
        };
        let media = MediaFacts {
            num_steps: 8,
            video_denoiser: None,
            state_bytes: 1000,
            local_product_bytes: 4096,
            latent: None,
            stores_latent: false,
            artifact_import_regions: 0,
        };
        let requirements = slot_requirements(&config, 12, &media, false, 2, 4).unwrap();
        assert_eq!(requirements.len(), 3);
        assert!(requirements.windows(2).all(|pair| pair[0] < pair[1]));
        let plan = request_layout(
            report(),
            &WorkerConfig {
                max_request_pool_size: 3,
                ..config
            },
            12,
            &media,
            false,
        )
        .unwrap();
        // Twelve positions over three slots keep three unresolved outputs.
        assert_eq!(plan.info.max_unresolved_calls, 3);
        assert_eq!(plan.info.buffer_pool_bytes, 3 * 4096);
        assert_eq!(plan.info.num_inference_steps, 8);
    }

    #[test]
    fn every_rank_of_a_sampling_worker_reports_the_latent_pool() {
        let config = WorkerConfig {
            max_batch_calls: 2,
            max_request_pool_size: 2,
            ..cpu_config()
        };
        // Two slots of eight pages each, and the sentinel page.
        let pool = LatentPlan {
            request_pool_size: 2,
            num_pages: 17,
            page_units: 1024,
            latent_width: 1,
            dtype: crate::ffi::DType::F32,
            with_workspace: false,
        };
        let facts = |stores_latent| MediaFacts {
            num_steps: 8,
            video_denoiser: None,
            state_bytes: 1000,
            local_product_bytes: 4096,
            latent: Some(pool),
            stores_latent,
            artifact_import_regions: 0,
        };
        let layout =
            |stores_latent| request_layout(report(), &config, 12, &facts(stores_latent), false);
        let denoising = layout(true).unwrap();
        let other = layout(false).unwrap();

        // The head compares the ranks' reports and schedules pages by them.
        assert_eq!(other.info, denoising.info);
        assert_eq!(denoising.info.latent_pages, 17);
        assert_eq!(denoising.info.latent_page_units, 1024);

        // Only the denoising rank stores the pool.
        assert_eq!(denoising.latent, Some(pool));
        assert_eq!(other.latent, None);
        let requirement = |stores_latent| {
            slot_requirements(&config, 12, &facts(stores_latent), false, 2, 2).unwrap()[0]
        };
        assert!(requirement(false) < requirement(true));
    }
}

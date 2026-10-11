//! Startup capacity planning: page size, request slots, KV units, latent
//! pages and the [`WorkerInfo`] the rank reports.
//!
//! Planning runs after the model is loaded, so measured device storage
//! includes the weights. Every quantity the engine schedules against must be
//! identical on the ranks of a group, so the page size, the request-slot fit
//! and the KV unit count are agreed over the group's [`HostGroup`] with a
//! minimum reduction.
//!
//! Token workers (text, canvas, image) place a KV pool beside fixed storage:
//! the buffer pool, the tensor store's products, the latent pool and the
//! storage each capability runtime reserves (input buffers, block tables,
//! decode state, importer scratch). Automatic KV sizing gives the pool the
//! grant left after those and a graph-capture reservation. Media workers
//! (video) size request slots instead, because every resident request keeps
//! its whole media state.

use std::collections::BTreeMap;

use std::sync::Arc;

use uniserve_ipc::{
    CallKind, ComponentInfo, DType, MediaCall, OutputInfo, VideoDenoiserInfo, WorkerEndpoint,
    WorkerInfo,
};

use super::{
    ArenaCapacity, DEFAULT_NUM_UNITS, KVCapacity, call_window, latent_pool_bytes,
    request_tensor_window,
};
use crate::config::{Device, WorkerConfig, graph_padding_block_count};
use crate::cuda_graph::graph_storage_budget_bytes;
use crate::parallel::host::HostGroup;
use crate::storage::KVPlan;
use crate::{Error, Result};

/// Product scalar banks keep one value of every protocol dtype per slot.
pub fn product_scalar_bytes() -> u64 {
    [
        DType::U8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::F16,
        DType::BF16,
        DType::F32,
    ]
    .iter()
    .map(|dtype| dtype.element_bytes())
    .sum()
}

/// Bytes one product slot of a component's outputs retains: every output's
/// largest extent, aligned to 256 bytes.
pub fn product_bytes(outputs: &[OutputInfo]) -> u64 {
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
pub fn page_size(
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
pub fn request_slots(
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

/// Device dimensions of one latent pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatentPlan {
    pub request_pool_size: usize,
    /// Pages including the reserved sentinel page.
    pub num_pages: usize,
    pub page_units: usize,
    pub latent_width: usize,
    pub element_bytes: usize,
    /// A step workspace beside the two banks (image denoising).
    pub with_workspace: bool,
}

impl LatentPlan {
    /// The pool of an image flow: pages of `block_size` units holding the
    /// longest image (or the configured token budget, when larger).
    pub fn image(
        slots: usize,
        max_tokens: usize,
        kv_token_capacity: Option<usize>,
        block_size: usize,
        width: usize,
        element_bytes: usize,
    ) -> Self {
        let units = if max_tokens == 0 {
            0
        } else {
            max_tokens.max(kv_token_capacity.unwrap_or(max_tokens))
        };
        Self {
            request_pool_size: slots,
            num_pages: units.div_ceil(block_size) + 1,
            page_units: block_size,
            latent_width: width,
            element_bytes,
            with_workspace: true,
        }
    }

    pub fn bytes(&self) -> Result<u64> {
        latent_pool_bytes(
            self.request_pool_size,
            self.num_pages,
            self.page_units,
            self.latent_width,
            self.element_bytes,
            self.with_workspace,
        )
    }
}

/// Model-independent values of the report.
pub struct Report {
    pub model_name: String,
    pub endpoint: WorkerEndpoint,
    pub device: String,
    pub transfer_backends: Vec<String>,
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
pub struct MediaFacts {
    pub num_steps: u32,
    pub video_denoiser: Option<VideoDenoiserInfo>,
    /// Device bytes of one request's state buffers.
    pub state_bytes: u64,
    /// Product bytes this rank retains per request.
    pub local_product_bytes: u64,
    /// Product bytes of every component per request (the logical pool).
    pub product_bytes: u64,
    /// Latent pool of sampled trajectories, when this rank holds samples.
    pub latent: Option<LatentPlan>,
    /// Artifact regions imported from other codec ranks.
    pub artifact_import_regions: usize,
}

/// Inputs of the token layout taken from the loaded model.
pub struct TokenFacts {
    /// The model has an image processor (it encodes request images).
    pub image_processor: bool,
    /// Bytes of one encoded image's vision features.
    pub vision_feature_bytes: u64,
    /// Bytes of one image's latent features (image flow).
    pub latent_feature_bytes: u64,
    /// The latent pool of an image flow.
    pub latent: Option<LatentPlan>,
    /// Aligned product bytes per request over every component.
    pub product_bytes: u64,
}

/// The resolved capacity of a rank.
pub struct Plan {
    pub info: WorkerInfo,
    pub arena: ArenaCapacity,
    /// Fixed bytes by device, beside the KV pool.
    pub fixed_device_bytes: BTreeMap<Device, u64>,
    /// Bytes of this rank's physical buffer pool.
    pub physical_buffer_pool_bytes: u64,
    pub latent: Option<LatentPlan>,
    /// The KV layout, with the agreed unit count in `info.kv_cache`.
    pub cache: Option<Arc<KVPlan>>,
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
        self.info.max_prefill_calls = self.info.max_prefill_calls.min(self.info.max_batch_calls);
        self.info.max_decode_calls = self.info.max_decode_calls.min(self.info.max_batch_calls);
    }

    /// Bound calls by the captured graphs a token runtime holds. Zero leaves
    /// a kind bounded by `max_batch_calls` alone.
    pub fn bound_graph_calls(&mut self, prefill: u32, decode: u32) {
        self.info.max_prefill_calls = prefill.min(self.info.max_batch_calls);
        self.info.max_decode_calls = decode.min(self.info.max_batch_calls);
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
        max_prefill_calls: 0,
        max_decode_calls: 0,
        request_slots: narrow(config.max_request_pool_size, "request slots")?,
        kv_cache: None,
        latent_page_units: 0,
        latent_pages: 0,
        buffer_pool_bytes: 0,
        encoder_cache_entries: 0,
        encoder_entry_bytes: 0,
        max_unresolved_calls: 0,
        host_lane_capacity: 1,
    })
}

/// A token worker's layout before the KV pool is fitted.
pub struct TokenLayout {
    info: WorkerInfo,
    arena: ArenaCapacity,
    fixed: BTreeMap<Device, u64>,
    feature_bytes: u64,
    latent: Option<LatentPlan>,
}

impl TokenLayout {
    /// Size the buffer pool, product store and latent pool.
    pub fn new(
        report: Report,
        config: &WorkerConfig,
        queue_depth: usize,
        facts: &TokenFacts,
    ) -> Result<Self> {
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
        info.buffer_pool_bytes = (u64::from(info.encoder_cache_entries) + 1) * feature_bytes
            + u64::from(info.request_slots) * facts.product_bytes;
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
        let arena = ArenaCapacity::for_tokens(
            queue_depth,
            config.max_batch_calls,
            config.max_request_pool_size,
            product_scalar_bytes(),
            latent_bytes,
            feature_bytes,
        )?;
        let mut fixed = BTreeMap::new();
        fixed.insert(
            config.device,
            info.buffer_pool_bytes + arena.device_product_bytes + latent_bytes,
        );
        info.host_lane_capacity = narrow(arena.host_lane_inflight, "host lane capacity")?;
        Ok(Self {
            info,
            arena,
            fixed,
            feature_bytes,
            latent: facts.latent,
        })
    }

    pub fn info(&self) -> &WorkerInfo {
        &self.info
    }

    /// Record fixed storage a capability runtime allocates on `device`.
    pub fn reserve(&mut self, device: Device, bytes: u64) {
        *self.fixed.entry(device).or_default() += bytes;
    }

    /// Fit the KV pool beside the fixed storage and agree its unit count.
    ///
    /// The pool takes the grant left after a graph-capture reservation (10%
    /// of the device), the widest default graph padding of every group, and
    /// the fixed storage, in whole units. A configured token capacity sizes
    /// it instead. `total_bytes` is the device's total memory, zero on CPU.
    /// `unit_overhead` is device storage other owners keep per unit (unit
    /// index columns), charged with each unit.
    pub fn fit_cache(
        &mut self,
        cache: &KVPlan,
        unit_overhead: u64,
        config: &WorkerConfig,
        total_bytes: u64,
        group: &HostGroup,
    ) -> Result<KVCapacity> {
        let unit_bytes = cache.unit_bytes + unit_overhead;
        let pages: Vec<(u32, u32)> = cache
            .groups
            .iter()
            .map(|group| (group.page_tokens, group.units_per_page))
            .collect();
        let mut reserved_units = 0;
        for &(tokens, units) in &pages {
            reserved_units += graph_padding_block_count(tokens as usize) as u64 * u64::from(units);
        }
        reserved_units += graph_storage_budget_bytes(total_bytes as i64).div_ceil(unit_bytes);
        reserved_units += self
            .fixed
            .get(&config.device)
            .copied()
            .unwrap_or(0)
            .div_ceil(unit_bytes);

        let cuda = config.device.is_cuda();
        let tokens = config.kv_token_capacity.map(|tokens| tokens as u64);
        let available = config.pool_storage_bytes.map(|bytes| bytes as u64);
        if cuda && tokens.is_none() && available.is_none() {
            return Err(Error::Invalid(
                "automatic CUDA KV sizing requires a host storage grant".into(),
            ));
        }
        let mut capacity = KVCapacity::derive(
            &pages,
            tokens,
            unit_bytes,
            available,
            1,
            DEFAULT_NUM_UNITS,
            1,
            reserved_units,
        )?;
        // Every member allocates the same logical unit ids, bounded by the
        // smallest grant.
        capacity.num_units = group.all_reduce_min(&[capacity.num_units as i64])?[0] as u64;
        let num_units = u32::try_from(capacity.num_units)
            .map_err(|_| Error::Invalid("KV pool exceeds the unit index range".into()))?;
        self.info.kv_cache = Some(cache.info(num_units));
        // The arena's slot counts are independent of KV capacity; only its
        // transfer byte reservation follows the fitted pool.
        self.arena.transfer_bytes = (capacity.num_units * cache.unit_bytes)
            .max(self.feature_bytes)
            .max(1)
            * self.arena.transfer_tickets as u64;
        Ok(capacity)
    }

    pub fn finish(self, config: &WorkerConfig, cache: Option<Arc<KVPlan>>) -> Plan {
        let physical_buffer_pool_bytes = self.info.buffer_pool_bytes;
        let mut plan = Plan {
            info: self.info,
            arena: self.arena,
            fixed_device_bytes: self.fixed,
            physical_buffer_pool_bytes,
            latent: self.latent,
            cache,
        };
        plan.clamp_lanes(config);
        plan
    }
}

/// The layout of a media worker, sized by its request slots.
pub fn request_layout(
    report: Report,
    config: &WorkerConfig,
    queue_depth: usize,
    media: &MediaFacts,
    host_components: bool,
) -> Result<Plan> {
    let mut info = base_info(report, config, queue_depth)?;
    let slots = info.request_slots as usize;
    info.max_unresolved_calls = narrow(
        request_tensor_window(queue_depth, slots)?,
        "unresolved calls",
    )?;
    info.max_batch_calls = info.max_batch_calls.min(info.request_slots);
    info.max_batch_tokens = info.max_batch_calls;
    info.buffer_pool_bytes = slots as u64 * media.product_bytes;
    info.num_inference_steps = media.num_steps;
    info.video_denoiser = media.video_denoiser.clone();
    if let Some(plan) = &media.latent {
        info.latent_page_units = narrow(plan.page_units, "latent page units")?;
        info.latent_pages = narrow(plan.num_pages, "latent pages")?;
    }
    let latent_bytes = media
        .latent
        .map(|plan| plan.bytes())
        .transpose()?
        .unwrap_or(0);
    let arena = ArenaCapacity::for_requests(
        queue_depth,
        info.max_batch_calls as usize,
        slots,
        config.world_size,
        product_scalar_bytes(),
        media.local_product_bytes,
        media.artifact_import_regions,
        latent_bytes,
    )?;
    info.host_lane_capacity = if host_components {
        1
    } else {
        narrow(arena.host_lane_inflight, "host lane capacity")?
    };
    let mut plan = Plan {
        info,
        arena,
        fixed_device_bytes: BTreeMap::new(),
        physical_buffer_pool_bytes: slots as u64 * media.local_product_bytes,
        latent: media.latent,
        cache: None,
    };
    plan.clamp_lanes(config);
    Ok(plan)
}

/// This rank's storage for every candidate slot count from `minimum` to
/// `maximum`: state, retained products, the product store and the latent
/// pool of each count.
pub fn slot_requirements(
    config: &WorkerConfig,
    queue_depth: usize,
    media: &MediaFacts,
    minimum: usize,
    maximum: usize,
) -> Result<Vec<u64>> {
    let mut requirements = Vec::new();
    for count in minimum..=maximum {
        let arena = ArenaCapacity::for_requests(
            queue_depth,
            config.max_batch_calls.min(count),
            count,
            config.world_size,
            product_scalar_bytes(),
            media.local_product_bytes,
            0,
            0,
        )?;
        let latent_bytes = match &media.latent {
            Some(plan) => {
                let pages = if plan.with_workspace {
                    plan.num_pages
                } else {
                    count * ((plan.num_pages - 1) / plan.request_pool_size) + 1
                };
                latent_pool_bytes(
                    count,
                    pages,
                    plan.page_units,
                    plan.latent_width,
                    plan.element_bytes,
                    plan.with_workspace,
                )?
            }
            None => 0,
        };
        requirements.push(
            count as u64 * (media.state_bytes + media.local_product_bytes)
                + arena.device_product_bytes
                + latent_bytes,
        );
    }
    Ok(requirements)
}

#[cfg(test)]
mod tests {
    use uniserve_ipc::{DimBound, ShapeBound};

    use crate::ffi::Fields;

    use super::*;

    fn group() -> HostGroup {
        HostGroup::single(0)
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
            fabric_handles: false,
            attention_backend: "auto".into(),
            supported_calls: vec![CallKind::Forward(uniserve_ipc::ForwardMode::Prefill)],
            media_components: BTreeMap::new(),
            components: Vec::new(),
            weight_formats: vec!["dense".into()],
            activation_formats: vec!["dense".into()],
        }
    }

    /// A two-layer, one-column bfloat16 pool with 64-token pages, as
    /// `model.cache_plan` describes it.
    fn cache() -> KVPlan {
        let ints = |values: &[i64]| crate::ffi::integer_list(values.iter().copied());
        let group = Fields::new()
            .with("window", -1i64)
            .with("page_tokens", 64i64)
            .with("units_per_page", 2i64)
            .with("layers", crate::ffi::strings(["layers.0", "layers.1"]))
            .with("layer_ids", ints(&[0, 1]))
            .with("axis_layers", ints(&[0, 1]))
            .with("num_kv_heads", 4i64)
            .with("total_kv_heads", 8i64)
            .with("kv_head_offset", 4i64)
            .with("head_dim", 128i64)
            .text("dtype", "bfloat16")
            .text("compute_dtype", "bfloat16")
            .into_any();
        let plane_bytes = 64 * 4 * 128 * 2;
        let plan = Fields::new()
            .with("columns", 1i64)
            .with("plane_bytes", plane_bytes)
            .with("quantized", false)
            .with("unit_bytes", 2 * (plane_bytes + 1))
            .text("dtype", "bfloat16")
            .with("groups", crate::ffi::list([group]))
            .into_any();
        KVPlan::decode(plan).unwrap()
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
        assert_eq!(product_scalar_bytes(), 23);
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
    fn the_kv_pool_takes_the_grant_left_after_reservations() {
        let gib = 1u64 << 30;
        let config = WorkerConfig {
            device: Device::Cuda(0),
            max_batch_calls: 8,
            max_batch_tokens: 2048,
            max_request_pool_size: 8,
            pool_storage_bytes: Some((20 * gib) as usize),
            ..WorkerConfig::default()
        };
        let facts = TokenFacts {
            image_processor: false,
            vision_feature_bytes: 0,
            latent_feature_bytes: 0,
            latent: None,
            product_bytes: 0,
        };
        let mut layout = TokenLayout::new(report(), &config, 2, &facts).unwrap();
        let fixed_before = layout.fixed[&Device::Cuda(0)];
        layout.reserve(Device::Cuda(0), gib);
        let cache = cache();
        let capacity = layout
            .fit_cache(&cache, 0, &config, 100 * gib, &group())
            .unwrap();

        let unit = cache.unit_bytes;
        let reserved = graph_padding_block_count(64) as u64 * 2
            + (10 * gib).div_ceil(unit)
            + (fixed_before + gib).div_ceil(unit);
        assert_eq!(capacity.num_units, 20 * gib / unit - reserved);
        let plan = layout.finish(&config, Some(Arc::new(cache)));
        let info = plan.info.kv_cache.as_ref().unwrap();
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
            ..WorkerConfig::default()
        };
        let facts = TokenFacts {
            image_processor: true,
            vision_feature_bytes: 1024,
            latent_feature_bytes: 512,
            latent: Some(LatentPlan::image(8, 1000, None, 64, 12, 2)),
            product_bytes: 256,
        };
        let mut layout = TokenLayout::new(report(), &config, 2, &facts).unwrap();
        let capacity = layout.fit_cache(&cache(), 0, &config, 0, &group()).unwrap();
        assert_eq!(capacity.num_units, DEFAULT_NUM_UNITS);
        let plan = layout.finish(&config, Some(Arc::new(cache())));
        assert_eq!(plan.info.max_batch_calls, 4);
        assert_eq!(plan.info.encoder_entry_bytes, 1024);
        assert_eq!(plan.info.buffer_pool_bytes, 257 * 1024 + 8 * 256);
        // 1000 tokens in 64-unit pages, plus the sentinel page.
        assert_eq!(plan.info.latent_pages, 17);
        assert_eq!(plan.info.latent_page_units, 64);
    }

    #[test]
    fn slot_requirements_grow_with_the_slot_count() {
        let config = WorkerConfig {
            max_batch_calls: 4,
            ..WorkerConfig::default()
        };
        let media = MediaFacts {
            num_steps: 8,
            video_denoiser: None,
            state_bytes: 1000,
            local_product_bytes: 4096,
            product_bytes: 8192,
            latent: None,
            artifact_import_regions: 0,
        };
        let requirements = slot_requirements(&config, 12, &media, 2, 4).unwrap();
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
        assert_eq!(plan.info.buffer_pool_bytes, 3 * 8192);
        assert_eq!(plan.physical_buffer_pool_bytes, 3 * 4096);
        assert_eq!(plan.info.num_inference_steps, 8);
    }
}

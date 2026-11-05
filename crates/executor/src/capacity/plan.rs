//! Startup capacity planning: block size, request slots, cache units, latent
//! pages and the [`WorkerInfo`] the rank reports.
//!
//! Planning runs after the model is loaded, so measured device storage
//! includes the weights. Every quantity the engine schedules against must be
//! identical on the ranks of a group, so the block size, the request-slot fit
//! and the cache unit count are agreed over the group's [`HostGroup`] with a
//! minimum reduction.
//!
//! Token workers (text, diffusion LM, image) place a cache pool beside fixed
//! storage: the buffer pool, the tensor store's products, the latent pool and
//! the storage the executor's computations and the prefix cache reserve (the
//! bytes the executor's manifest states, block tables, importer scratch).
//! Automatic cache sizing gives the pool the grant left after those and a
//! graph-capture reservation. Media workers (video) size request slots instead,
//! because every resident request keeps its whole media state.

use std::collections::BTreeMap;

use std::sync::Arc;

use ipc::{
    CallKind, ComponentInfo, MediaCall, OutputInfo, VideoDenoiserInfo, WorkerEndpoint, WorkerInfo,
};

use super::{
    ArenaCapacity, CacheCapacity, CacheDemand, RequestBounds, TokenBounds, call_window,
    request_tensor_window,
};
use crate::config::Device;
use crate::memory::prefix_cache::planes::{PlaneLayout, UnitColumn};
use crate::memory::{CachePlan, LatentPlan};

/// Unit columns the computations over the prefix cache bind beside the import
/// workers': one, on the device's primary stream.
const COMPUTATION_COLUMNS: usize = 1;
use crate::parallel::host::HostGroup;
use crate::{Error, Result};

/// The launch values capacity plans with, built by the engine from the launch
/// configuration it already reads.
///
/// Capacity is a pure computation over these: nothing here is a policy of its
/// own. The bucket sizes are the graph plan's, the device is where the rank
/// runs, and the pool and segment values are the deployment's cache decision.
#[derive(Clone, Debug, PartialEq)]
pub struct CapacityRequest {
    /// The device this rank serves.
    pub device: Device,
    /// Tokens per logical cache page, when the deployment fixed it.
    pub block_size: Option<usize>,
    /// Bytes the rank's share of the device grants for the cache pool.
    pub pool_storage_bytes: Option<usize>,
    /// The bytes it grants once nothing else holds the device.
    pub pool_storage_full_bytes: Option<usize>,
    /// Fraction of the device's total storage the process may hold.
    pub mem_fraction_static: f64,
    /// Most calls one batch may carry.
    pub max_batch_calls: usize,
    /// Most tokens one batch may carry.
    pub max_batch_tokens: usize,
    /// Most tokens a request may hold, when the deployment fixed it.
    pub max_total_tokens: Option<usize>,
    /// Request slots the rank keeps.
    pub max_request_pool_size: usize,
    /// Encoder cache entries the rank keeps.
    pub encoder_cache_entries: usize,
    /// The exact cache segments the rank's planes are made of.
    pub segments: Option<ipc::CacheTable>,
    /// Whether warmup storage is mapped writable and the pool's remaining
    /// virtual addresses are reserved for staged transfer.
    pub stage_cache: bool,
    /// The predecessor world's cache, whose segments the planes map read-only.
    pub predecessor: Option<ipc::PredecessorCache>,
    /// This rank's world.
    pub world: u64,
    /// Ranks of the deployment.
    pub world_size: usize,
    /// Decode graph batch sizes, from the graph plan.
    pub decode_graph_batch_sizes: Vec<usize>,
    /// Prefill graph token sizes, from the graph plan.
    pub prefill_graph_token_sizes: Vec<usize>,
    /// Flow graph shapes `(tokens, latent)` the media graph plan captures.
    pub flow_graph_shapes: Vec<(usize, usize)>,
}

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

/// The largest KV block size the rank resolves to unless the launch
/// presets one ([`resolve_block_size`]). It also bounds the latent rows of
/// an image denoiser's latent page, one KV block of them
/// ([`LatentPlan::image`]), by which the executor sizes its denoising
/// workspace (`uniserve.executor.image._MAX_PAGE_UNITS`).
pub(crate) const MAX_BLOCK_SIZE: usize = 64;

/// The largest block size of at most [`MAX_BLOCK_SIZE`] tokens that every
/// resident layer's attention reads, agreed over `group` as the minimum.
///
/// A preset size is used unchanged and needs no agreement: every rank was
/// launched with it. `readable` plans the pool for one candidate size.
pub(crate) fn resolve_block_size(
    preset: Option<usize>,
    mut readable: impl FnMut(usize) -> Result<bool>,
    group: &HostGroup,
) -> Result<usize> {
    if let Some(size) = preset {
        return Ok(size);
    }
    let mut size = MAX_BLOCK_SIZE;
    while !readable(size)? {
        size /= 2;
        if size == 0 {
            return Err(Error::Unsupported(format!(
                "no KV block size of at most {MAX_BLOCK_SIZE} tokens is read by the attention of every resident cache layer"
            )));
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
            // A slot holds one request's tensors at the deployment's
            // capacities, so the operator either serves smaller ones or
            // grants the storage.
            Error::Unsupported(format!(
                "insufficient device storage for a common request tensor slot count: candidate range {minimum}..{}, local requirements {requirements:?}, {available} bytes available; serve smaller prompt, condition or duration capacities, or give the worker more device storage (--mem-fraction-static)",
                minimum + requirements.len() - 1,
            ))
        })
}

impl LatentPlan {
    /// The pool of an image flow: pages of `block_size` units holding the
    /// longest image (or the configured token budget, when larger). The
    /// executor keeps the step workspace its forwards read beside it, which
    /// it sizes for pages of at most [`MAX_BLOCK_SIZE`] units: a larger
    /// block size is refused.
    pub(crate) fn image(
        slots: usize,
        max_tokens: usize,
        max_total_tokens: Option<usize>,
        block_size: usize,
        width: usize,
        dtype: crate::memory::allocation::ElementType,
    ) -> Result<Self> {
        if block_size > MAX_BLOCK_SIZE {
            return Err(Error::Unsupported(format!(
                "an image denoiser's latent page holds one KV block of at most \
                 {MAX_BLOCK_SIZE} latent rows, which its executor sizes its denoising \
                 workspace by; kv_cache_request.block_size {block_size} exceeds it"
            )));
        }
        let units = if max_tokens == 0 {
            0
        } else {
            max_tokens.max(max_total_tokens.unwrap_or(max_tokens))
        };
        Ok(Self {
            request_pool_size: slots,
            num_pages: units.div_ceil(block_size) + 1,
            page_units: block_size,
            latent_width: width,
            dtype,
        })
    }
}

/// Model-independent values of the report.
pub struct Report {
    pub endpoint: WorkerEndpoint,
    /// The rank publishes device products over CUDA VMM, which keeps an
    /// export pool on its primary device.
    pub device_exports: bool,
    pub fabric_handles: bool,
    pub supported_calls: Vec<CallKind>,
    pub media_components: BTreeMap<MediaCall, String>,
    pub components: Vec<ComponentInfo>,
}

/// Media-timeline facts of a video model.
#[derive(Debug, Clone, PartialEq)]
pub struct MediaFacts {
    pub video_denoiser: Option<VideoDenoiserInfo>,
    /// Device bytes of one request's state buffers.
    pub state_bytes: u64,
    /// Product bytes this rank backs per request: those of the components it
    /// executes and of the components it reads.
    pub local_product_bytes: u64,
    /// Latent pool of sampled trajectories, when the worker holds the
    /// denoiser. Every rank of the worker reports its pages, because the engine
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
pub struct TokenFacts {
    /// The model has an image processor (it encodes request images).
    pub image_processor: bool,
    /// Bytes of one encoded image's vision features.
    pub vision_feature_bytes: u64,
    /// Bytes of one image's latent features (image flow).
    pub latent_feature_bytes: u64,
    /// The latent pool of an image flow.
    pub latent: Option<LatentPlan>,
}

/// The resolved capacity of a rank: what the main process builds its resident
/// memory from.
pub struct Plan {
    /// The capacity the rank reports to the engine.
    pub info: WorkerInfo,
    /// The host and device arenas of its calls: read where a rank's arenas are
    /// built, which is inside this crate.
    pub(crate) arena: ArenaCapacity,
    /// Fixed bytes by device, beside the cache pool.
    pub fixed_device_bytes: BTreeMap<Device, u64>,
    /// The latent pool of a rank that stores one.
    pub latent: Option<LatentPlan>,
    /// The cache layout, with the agreed unit count in `info.prefix_cache`.
    pub cache: Option<Arc<CachePlan>>,
    /// The segments a device pool's planes are made of; `None` for planes
    /// that do not split.
    pub planes: Option<PlaneLayout>,
    /// Writable units used by the executor's largest warmup phase.
    pub warmup_units: u32,
}

fn narrow(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::Invalid(format!("{name} exceeds the protocol range")))
}

fn base_info(report: Report, request: &CapacityRequest) -> Result<WorkerInfo> {
    Ok(WorkerInfo {
        media_components: report.media_components,
        video_denoiser: None,
        endpoint: report.endpoint,
        fabric_handles: report.fabric_handles,
        components: report.components,
        supported_calls: report.supported_calls,
        max_batch_calls: narrow(request.max_batch_calls, "max batch calls")?,
        max_batch_tokens: narrow(request.max_batch_tokens, "max batch tokens")?,
        request_slots: narrow(request.max_request_pool_size, "request slots")?,
        prefix_cache: None,
        latent_page_units: 0,
        latent_pages: 0,
        // A video denoising rank maps its host latent pool once its ledger
        // is bound.
        latent_pool: None,
        encoder_pool: None,
        // An image encoding rank states its encoders once its executor
        // stated them.
        image_encoders: Box::default(),
        buffer_pool_bytes: 0,
        encoder_cache_entries: 0,
        encoder_entry_bytes: 0,
        max_unresolved_calls: 0,
        host_task_capacity: 1,
        transfer_tickets: 1,
        image_warmups: Vec::new(),
        captured_images: None,
        captured_latents: None,
        image_capacity: None,
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
    planes: Option<PlaneLayout>,
    warmup_units: u32,
}

/// The three numbers a dense cache's geometry is stated by, as
/// [`CacheDescription::dense`](crate::memory::CacheDescription::dense)
/// takes them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheGeometry {
    pub layers: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
}

/// Everything the engine states about the rank it starts.
///
/// Every figure here is one the engine owns or has already agreed; nothing is
/// inferred, because a wrong admission bound or memory fraction is a silently
/// smaller cache rather than an error.
pub struct RankFacts<'a> {
    pub device: Device,
    /// The rank's host endpoint, as it advertises itself.
    pub host: &'a str,
    pub world: u64,
    /// The deployment's rank count, which no peer list states.
    pub world_size: usize,
    /// Requests the scheduler may have in flight, which the daemon places.
    pub queue_depth: usize,
    /// Tokens per logical cache page, when the deployment fixed it.
    pub block_size: Option<u32>,
    /// Most calls and most tokens one batch may carry: the largest bucket of
    /// the rank's catalog, which is its admission bound because a larger batch
    /// is refused before a kernel runs.
    pub max_batch_calls: usize,
    pub max_batch_tokens: usize,
    /// Request slots the rank keeps.
    pub max_request_pool_size: usize,
    /// The model's longest request.
    pub max_model_len: usize,
    /// Fraction of the device's storage the epoch may hold.
    pub mem_fraction_static: f64,
    /// The paged key/value geometry this rank keeps, or `None` for a rank that keeps no paged
    /// cache at all - a media rank, whose resident state is the request's own whole footprint
    /// rather than pages a scheduler addresses. `plan_rank` plans the first and `plan_media` the
    /// second, and each refuses the other's value by name: a plan fitted for one is not a plan the
    /// other can honour, and the two would differ silently if a zero stood for absence.
    pub cache: Option<CacheGeometry>,
    pub kv_dtype: ipc::PrefixCacheDtype,
}

/// A rank's plan with the figures no owner states: the widest block table a
/// request needs, which follows from the block size the plan chose, and the two
/// reservations the rank takes out of its grant before a cache unit is fitted -
/// the graph range and the shadow region beside it - which the fit takes as an
/// input and would otherwise not be readable back from the plan.
pub struct TokenPlan {
    pub plan: Plan,
    pub width: u32,
    /// The graph range the rank reserves: one pool per generation a preparation
    /// may hold, mapped whole at the first capture.
    pub reserved: u64,
    /// The shadow region a replaced operator runs in, reserved beside it.
    pub shadow: u64,
}

/// One region of device storage a rank's plan books (design 4.1 and 4.5).
///
/// The list is the plan's own: a region a plan books and this enum does not name
/// is a region no reader can compare against a published gauge. `slot` matches
/// every variant, so adding one here without adding it to `Booked` is a compile
/// error rather than a shortfall in a sum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Region {
    /// The checkpoint's tensors, mapped before the plan runs and booked by the
    /// Storage Server's device ledger.
    Weights,
    /// The fitted cache pool: `units x unit_bytes`, mapped through the cache's
    /// own planes and booked by the Storage Server's ledger as cache segments.
    Cache,
    /// The graph range, reserved whole and never a bucket at a time.
    Graph,
    /// The shadow region a replaced operator is executed in.
    Shadow,
    /// The state snapshot a unit's checkpoints are copied through.
    Snapshot,
    /// Everything else the resident owners map: block tables, request-slot banks,
    /// import scratch, the buffer pool, the product store's deferred arenas and
    /// the latent pool.
    Fixed,
}

impl Region {
    /// Every region a plan books, in the order a report reads them.
    pub const ALL: [Region; Region::COUNT] = [
        Region::Weights,
        Region::Cache,
        Region::Graph,
        Region::Shadow,
        Region::Snapshot,
        Region::Fixed,
    ];

    /// How many regions there are. `ALL` is an array of exactly this length, so a
    /// new variant breaks both this count and `slot`'s match.
    pub const COUNT: usize = 6;

    /// The region's index in [`Booked`].
    ///
    /// Exhaustive on purpose: a region added to the enum and not to this match
    /// does not compile.
    fn slot(self) -> usize {
        match self {
            Region::Weights => 0,
            Region::Cache => 1,
            Region::Graph => 2,
            Region::Shadow => 3,
            Region::Snapshot => 4,
            Region::Fixed => 5,
        }
    }
}

/// What a plan books of a device, region by region, and its total.
///
/// This is the plan's own sum and it is deliberately more than any one gauge
/// publishes:
///
/// - **Weights and Cache are not in the allocator's gauge.** They are mapped
///   through the Storage Server and booked in the GPU's device ledger, a
///   different owner; `uniserve:memory_mapped` counts the allocator's three pools
///   and `uniserve:memory_budget` is that allocator's allowance, so the gauge's
///   two figures are a local invariant and never this total.
/// - **The pinned pool is not a region at all.** Page-locked host blocks are host
///   bytes, outside every device sum by construction; counting them here would be
///   the opposite error.
/// - **Graph is the reservation, not a bucket.** The graph pool is mapped whole at
///   the first capture, so the figure a budget carries is the whole range and
///   never the widest bucket inside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Booked([u64; Region::COUNT]);

impl Booked {
    /// The bytes booked for `region`.
    pub fn get(&self, region: Region) -> u64 {
        self.0[region.slot()]
    }

    /// Every region with its bytes, in the order of [`Region::ALL`].
    pub fn regions(&self) -> impl Iterator<Item = (Region, u64)> + '_ {
        Region::ALL
            .into_iter()
            .map(|region| (region, self.get(region)))
    }

    /// The plan's whole booked sum on this device.
    pub fn total(&self) -> u64 {
        self.0.iter().sum()
    }
}

impl TokenPlan {
    /// The device storage this plan books of `device`, region by region.
    ///
    /// `weights` is the checkpoint's mapped bytes, which the plan cannot know:
    /// the store maps them before the plan measures the device, and its figure is
    /// what the fit's grant already excludes.
    pub fn booked(&self, device: Device, weights: u64) -> Booked {
        let cache = self
            .plan
            .cache
            .as_ref()
            .map(|cache| {
                let units = self
                    .plan
                    .info
                    .prefix_cache
                    .as_ref()
                    .map_or(0, |info| u64::from(info.num_units));
                units * cache.unit_bytes
            })
            .unwrap_or(0);
        let mut bytes = [0_u64; Region::COUNT];
        bytes[Region::Weights.slot()] = weights;
        bytes[Region::Cache.slot()] = cache;
        bytes[Region::Graph.slot()] = self.reserved;
        bytes[Region::Shadow.slot()] = self.shadow;
        bytes[Region::Snapshot.slot()] = self.state_snapshot_bytes();
        bytes[Region::Fixed.slot()] = self
            .plan
            .fixed_device_bytes
            .get(&device)
            .copied()
            .unwrap_or(0);
        Booked(bytes)
    }

    /// Bytes one unit's state snapshot occupies, which the scratch pool has to hold beside the
    /// shadow region (design 4.5): the cache plan's own figure, and zero when the model keeps no
    /// state per unit.
    pub fn state_snapshot_bytes(&self) -> u64 {
        self.plan
            .cache
            .as_ref()
            .map_or(0, |cache| cache.state_snapshot_bytes())
    }

    /// The tokens one page of the served cache holds.
    pub fn block_size(&self) -> u32 {
        self.plan
            .cache
            .as_ref()
            .and_then(|cache| cache.groups.first().map(|group| group.block_size))
            .unwrap_or(0)
    }
}

/// Resolve a token rank's capacity plan: the buffers, products and latents
/// sized, the block size agreed over `group`, the cache pool fitted to the
/// device grant.
///
/// `cache` is the model's geometry: called with a candidate block size, it
/// answers that size's plan, or `None` when the model cannot serve it. It is
/// called at least once per candidate size until the ranks agree, so a caller
/// memoizes it.
///
/// `width` is the widest block table a request of the model's length needs,
/// which the caller derives from its own `max_model_len`; `total_bytes` is the
/// storage the rank may plan into - what the device has left at plan time, not
/// the size it has, because the weights are mapped outside every pool - zero on
/// CPU, which states no grant; `reserved_bytes` is the storage the rank maps
/// beside the cache and out of its own grant, so that no unit is fitted into it:
/// the graph range and the scratch region, which `Pools::new` reserves as
/// separate pools; `granularity` is the device's allocation granularity, in
/// whose granules a device pool splits its planes.
#[expect(
    clippy::too_many_arguments,
    reason = "the request, the measured report, the model's facts and the agreement"
)]
pub fn token_plan(
    request: &CapacityRequest,
    report: Report,
    facts: &TokenFacts,
    queue_depth: usize,
    width: u32,
    total_bytes: u64,
    reserved_bytes: u64,
    granularity: Option<u64>,
    group: &HostGroup,
    mut cache: impl FnMut(usize) -> Result<Option<Arc<CachePlan>>>,
) -> Result<Plan> {
    let mut layout = TokenLayout::new(report, request, queue_depth, facts)?;
    let block_size =
        resolve_block_size(request.block_size, |size| Ok(cache(size)?.is_some()), group)?;
    let plan = cache(block_size)?.ok_or_else(|| {
        Error::Invariant(format!(
            "the model served block size {block_size} while the ranks agreed it"
        ))
    })?;

    // The cache's fixed storage beside its units: block tables, import scratch
    // and the request-slot banks. Every unit index column is charged with each
    // unit instead, because the unit count is not known until the pool is
    // fitted - so no column may be charged here as well.
    let fixed = crate::memory::prefix_cache::PrefixCache::fixed_bytes(
        &plan,
        0,
        layout.info().request_slots,
        width,
        layout.info().max_unresolved_calls as usize,
        0,
    )?;
    layout.reserve(request.device, fixed);

    // Every unit index column holds one index per unit and is charged with each
    // unit: the import workers' columns and the streams the computations over
    // the cache bind. Charging them here and not in the fixed figure is what
    // keeps a unit's bytes the same whether the pool is fitted before or after
    // its size is known.
    let workers = crate::memory::prefix_cache::PrefixCache::import_workers(
        layout.info().max_unresolved_calls as usize,
    ) as u64;
    let unit_overhead = (workers + COMPUTATION_COLUMNS as u64) * UnitColumn::bytes(1);
    layout.fit_cache(
        &plan,
        unit_overhead,
        request,
        total_bytes,
        reserved_bytes,
        granularity,
        group,
    )?;
    Ok(layout.finish(Some(plan)))
}

/// Resolve a token rank's plan from what the engine states, agreeing the block
/// size and the slot count over `group` and fitting the cache pool to the
/// device's grant.
///
/// The cache geometry is the model's own three numbers: the plan's page size
/// may come out lower than the requested one, because every layer's attention
/// must be able to read it, and the answer states the width a batch's block
/// table needs at that page size.
pub fn plan_rank(facts: &RankFacts<'_>, report: Report, group: &HostGroup) -> Result<TokenPlan> {
    let cache = facts.cache.ok_or_else(|| {
        Error::Invalid(
            "this rank states no paged key/value geometry, so no cache can be planned for it: \
             `plan_media` is the planner for a rank that keeps none"
                .into(),
        )
    })?;
    plan(facts, report, group, Some(cache))
}

/// What a rank that keeps **no paged key/value cache** plans: the same regions, no cache fitted.
///
/// A media rank's resident state is each request's own whole footprint - a layout's conditioning
/// rows, its latent trajectory and its clip - not pages a scheduler addresses, so there is nothing
/// to fit and nothing to divide into units. What it does book is what every rank books: the graph
/// storage the captures reserve, the scratch region the shadow and the state group's snapshot share,
/// and the fixed storage the rank's owners allocate. The plan's `cache` and the worker info's
/// `prefix_cache` are then absent, which is what a scheduler reads to know this rank takes no
/// paged request.
pub fn plan_media(facts: &RankFacts<'_>, report: Report, group: &HostGroup) -> Result<TokenPlan> {
    if facts.cache.is_some() {
        return Err(Error::Invalid(
            "this rank states a paged key/value geometry and the media class keeps no paged \
             cache: `plan_rank` is the planner for a rank that keeps one, and a geometry here \
             would fit a cache nothing addresses"
                .into(),
        ));
    }
    plan(facts, report, group, None)
}

/// The one planner both entry points share: what differs between them is only the cache fitted.
fn plan(
    facts: &RankFacts<'_>,
    report: Report,
    group: &HostGroup,
    cache: Option<CacheGeometry>,
) -> Result<TokenPlan> {
    // What the rank may plan into is what is LEFT: the weights are already on the device, so the
    // pools fit into the free bytes and not into the device's total.
    let total_bytes = match facts.device {
        Device::Cuda(index) => crate::capacity::device::available_bytes(index)?,
        Device::Cpu => 0,
    };
    // The graph storage is the one figure that is a share of the DEVICE and not of what is left: a
    // rank reserves two pools of address space at open, one per generation a preparation may hold,
    // and books their mappings against the device's total.
    // The graph range and the scratch region are one reservation (4.1, 4.5): the graph pools the
    // rank captures into and the shadow a replaced operator runs in, both reserved at spawn and
    // both out of the grant the cache is fitted into. `reserved_bytes` answers them as one figure
    // so a caller cannot pass half of it - a plan fitted into the bytes the shadow will take leaves
    // the rank to map past the device at its first shadow, after the model load and after every
    // module installed.
    // The two halves are also kept apart, because a reader of the finished plan
    // has to be able to compare each region with the pool that maps it: the graph
    // range is the graph pool's whole reservation, and the scratch region is the
    // shadow plus, for a model with a state group, the snapshot.
    let (reserved_bytes, graph_bytes) = match facts.device {
        Device::Cuda(index) => {
            let total = crate::capacity::device::total_bytes(index)?;
            let graph = crate::capacity::graph_storage_bytes(total as i64, true, false);
            (crate::capacity::reserved_bytes(total, true), graph)
        }
        Device::Cpu => (0, 0),
    };
    let request = CapacityRequest {
        device: facts.device,
        block_size: facts.block_size.map(|size| size as usize),
        pool_storage_bytes: None,
        pool_storage_full_bytes: None,
        mem_fraction_static: facts.mem_fraction_static,
        max_batch_calls: facts.max_batch_calls,
        max_batch_tokens: facts.max_batch_tokens,
        max_total_tokens: Some(facts.max_model_len),
        max_request_pool_size: facts.max_request_pool_size,
        encoder_cache_entries: 0,
        segments: None,
        stage_cache: false,
        predecessor: None,
        world: facts.world,
        world_size: facts.world_size,
        decode_graph_batch_sizes: Vec::new(),
        prefill_graph_token_sizes: Vec::new(),
        flow_graph_shapes: Vec::new(),
    };
    let token_facts = TokenFacts {
        image_processor: false,
        vision_feature_bytes: 0,
        latent_feature_bytes: 0,
        latent: None,
    };
    let plan = token_plan(
        &request,
        report,
        &token_facts,
        facts.queue_depth,
        0,
        total_bytes,
        reserved_bytes,
        None,
        group,
        |size| match &cache {
            Some(geometry) => Ok(Some(Arc::new(crate::memory::CachePlan::new(
                crate::memory::CacheDescription::dense(
                    size as u32,
                    geometry.layers,
                    geometry.kv_heads,
                    geometry.head_dim,
                    facts.kv_dtype,
                ),
            )?))),
            // No cache: the closure answers that the rank fits none, which is what makes the plan
            // state an absent cache rather than a cache of zero units.
            None => Ok(None),
        },
    )?;
    let mut answer = TokenPlan {
        plan,
        width: 0,
        reserved: graph_bytes,
        shadow: 0,
    };
    // The scratch region holds the shadow beside the snapshot the plan's own
    // `state_snapshot_bytes` answers, so the shadow is what the reservation has
    // left once the snapshot is taken out of it.
    answer.shadow = reserved_bytes - graph_bytes - answer.state_snapshot_bytes();
    let block = answer.block_size().max(1);
    answer.width = u32::try_from(facts.max_model_len.div_ceil(block as usize))
        .map_err(|_| Error::Invalid("the model's length states too many pages".into()))?;
    Ok(answer)
}

impl TokenLayout {
    /// Size the buffer pool, product store and latent pool.
    pub(crate) fn new(
        report: Report,
        request: &CapacityRequest,
        queue_depth: usize,
        facts: &TokenFacts,
    ) -> Result<Self> {
        let device_exports = report.device_exports;
        let mut info = base_info(report, request)?;
        let latent_bytes = facts
            .latent
            .map(|plan| plan.bytes())
            .transpose()?
            .unwrap_or(0);
        let feature_bytes = facts.latent_feature_bytes.max(facts.vision_feature_bytes);
        info.encoder_cache_entries = if facts.image_processor {
            narrow(request.encoder_cache_entries, "encoder cache entries")?
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
            call_window(queue_depth, request.max_batch_calls)?,
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
            calls: request.max_batch_calls,
            requests: request.max_request_pool_size,
            transfer_bytes_per_read: feature_bytes,
            device_exports,
        })?;
        let mut fixed = BTreeMap::new();
        fixed.insert(
            request.device,
            info.buffer_pool_bytes + arena.deferred_device_bytes() + latent_bytes,
        );
        info.host_task_capacity = narrow(arena.host_task_inflight, "host task capacity")?;
        info.transfer_tickets = narrow(arena.transfer_tickets, "transfer tickets")?;
        Ok(Self {
            info,
            arena,
            fixed,
            feature_bytes,
            latent: facts.latent,
            planes: None,
            warmup_units: 0,
        })
    }

    pub(crate) fn info(&self) -> &WorkerInfo {
        &self.info
    }

    /// Record fixed storage an owner of the rank allocates on `device`.
    pub(crate) fn reserve(&mut self, device: Device, bytes: u64) {
        *self.fixed.entry(device).or_default() += bytes;
    }

    /// Record the writable scratch capacity of the executor's warmup layouts.
    pub(crate) fn reserve_warmup(&mut self, units: u32) {
        self.warmup_units = units;
    }

    /// Fit the cache pool beside the fixed storage and agree its unit count.
    ///
    /// The pool takes the grant left after the storage the rank reserved
    /// (`reserved_bytes`: the graph range and the shadow region, and this plan's
    /// own state snapshot, charged here because the scratch holds one beside the
    /// other) and the fixed storage, in whole units. Every rank reserves that storage, whatever its
    /// graph policy: graph
    /// capture grows into it, and so do the eager allocations warmup makes
    /// (operator workspaces such as an expert kernel's), which a rank that
    /// serves every call eagerly makes too. Graph padding
    /// takes no unit: padding rows read the sentinel unit and write no K/V.
    /// A configured token capacity sizes the pool instead. `total_bytes` is
    /// what the device has left to plan into, zero on CPU. `reserved_bytes` is
    /// the storage the rank reserved - two pools of the device's 10% when a
    /// preparation may hold a second generation, plus the shadow region - which
    /// is what the pools reserve at open, so no unit is fitted into storage they
    /// will take; the state snapshot joins it below, when the plan holds one - and
    /// `unit_overhead` is device
    /// storage other owners keep per unit (unit index columns), charged with
    /// each unit; `total_bytes` is the storage the rank may plan into, the
    /// grant when the request states none. `granularity` is the device's
    /// allocation granularity, in
    /// whose granules a device pool splits its planes into segments; `None`
    /// on CPU.
    #[expect(
        clippy::too_many_arguments,
        reason = "the plan, the request, the device's figures and the agreement"
    )]
    pub(crate) fn fit_cache(
        &mut self,
        cache: &CachePlan,
        unit_overhead: u64,
        request: &CapacityRequest,
        total_bytes: u64,
        reserved_bytes: u64,
        granularity: Option<u64>,
        group: &HostGroup,
    ) -> Result<CacheCapacity> {
        let unit_bytes = cache.unit_bytes + unit_overhead;
        let pages: Vec<(u32, u32)> = cache
            .groups
            .iter()
            .map(|group| (group.block_size, group.units_per_page))
            .collect();
        // The shadow region and the state group's snapshot are one scratch pool (4.5): a twin
        // allocated from the shadow and a unit's checkpoint copied through the snapshot are
        // mapped beside the graph range, so the cache is fitted past neither.
        let share = reserved_bytes.saturating_add(cache.state_snapshot_bytes());
        let reserved_units = share.div_ceil(unit_bytes)
            + self
                .fixed
                .get(&request.device)
                .copied()
                .unwrap_or(0)
                .div_ceil(unit_bytes);

        let cuda = request.device.is_cuda();
        let tokens = request.max_total_tokens.map(|tokens| tokens as u64);
        // The grant the pool is fitted into: the host storage a deployment states for it, or -
        // when it states none - the storage the device has left, which is `total_bytes` here
        // because the weights are mapped by handle outside every pool and so never count against
        // a budget that would otherwise size the cache into bytes the weights already hold. A
        // device that states neither cannot be sized automatically, below.
        let available = request
            .pool_storage_bytes
            .map(|bytes| bytes as u64)
            .or(Some(total_bytes).filter(|bytes| *bytes > 0));
        if cuda && tokens.is_none() && available.is_none() {
            return Err(Error::Invalid(
                "automatic CUDA KV sizing requires a host storage grant".into(),
            ));
        }
        // A successor world maps its predecessor's segments beside units of
        // its own, which alone its grant pays for; a standby rank group's
        // worker adopts the segments of the worker it stands by for.
        let inherited: u64 = request.predecessor.as_ref().map_or(0, |predecessor| {
            predecessor
                .segments
                .iter()
                .map(|segment| u64::from(segment.units))
                .sum()
        });
        let adopted = request.segments.as_ref().map(|table| {
            table
                .segments
                .iter()
                .map(|segment| u64::from(segment.units))
                .sum::<u64>()
        });
        let mut capacity = CacheCapacity::derive(&CacheDemand {
            pages: &pages,
            token_units: tokens
                .map(|tokens| cache.token_units(tokens, request.max_request_pool_size as u64)),
            units: adopted,
            unit_bytes,
            available_bytes: available,
            reserved_units,
            // A successor's own units are what a configured capacity
            // leaves past the inherited ones, as far as its grant holds them
            // now, or what the grant holds.
            successor: request.predecessor.is_some() || request.stage_cache,
            inherited_units: inherited,
        })?;
        // Every member allocates the same logical unit ids, bounded by the
        // smallest grant.
        capacity.num_units = group.all_reduce_min(&[capacity.num_units as i64])?[0] as u64;
        // A device pool splits its value planes into segments of whole
        // allocation granules, which give storage back: its own units round
        // down to them, alike on every member. An adopted pool split
        // already.
        let mut planes = None;
        if let Some(granularity) = granularity {
            let own = u32::try_from(capacity.num_units)
                .map_err(|_| Error::Invalid("KV pool exceeds the unit index range".into()))?;
            planes = match (&request.segments, &request.predecessor) {
                (Some(table), _) => Some(PlaneLayout::adopted(table)),
                (None, predecessor) if predecessor.is_some() || request.stage_cache => {
                    Some(self.successor_layout(
                        cache,
                        request,
                        predecessor.as_ref(),
                        own,
                        unit_bytes,
                        reserved_units,
                        granularity,
                        group,
                    )?)
                }
                (None, _) => {
                    let rounded = crate::memory::prefix_cache::planes::segmented_units(
                        own,
                        cache.plane_bytes,
                        granularity,
                    );
                    tracing::info!(
                        fitted = own,
                        units = rounded,
                        "rounded the KV cache down to whole segments"
                    );
                    capacity.num_units = u64::from(rounded);
                    PlaneLayout::own(request.world, rounded, cache.plane_bytes, granularity)
                }
            };
            if let Some(layout) = &planes {
                capacity.num_units = u64::from(layout.units());
            }
        } else if inherited > 0 {
            return Err(Error::Invalid(
                "a host pool inherits no predecessor's segments".into(),
            ));
        }
        let num_units = u32::try_from(capacity.num_units)
            .map_err(|_| Error::Invalid("KV pool exceeds the unit index range".into()))?;
        let mut info = cache.info(num_units);
        if let Some(layout) = &planes {
            info.segments = layout.segments.iter().map(|laid| laid.segment).collect();
            info.held_segments = layout.segments.len() as u32;
            let inherited = layout.inherited();
            info.inherited_start = inherited.start;
            info.inherited_units = inherited.end - inherited.start;
        }
        self.info.prefix_cache = Some(info);
        self.planes = planes;
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

    /// A successor's writable head holds the sentinel, its executor's
    /// warmup scratch and request progress. Optional predecessor segments
    /// follow read-only, then the target's tail at its full grant. Host
    /// staging reserves the tail without creating it; other successors
    /// create as many segments as their present grant holds.
    #[expect(
        clippy::too_many_arguments,
        reason = "one successor pool's plan, its grants and its group"
    )]
    fn successor_layout(
        &self,
        cache: &CachePlan,
        request: &CapacityRequest,
        predecessor: Option<&ipc::PredecessorCache>,
        now: u32,
        unit_bytes: u64,
        reserved_units: u64,
        granularity: u64,
        group: &HostGroup,
    ) -> Result<PlaneLayout> {
        let inherited: u32 = predecessor
            .into_iter()
            .flat_map(|predecessor| &predecessor.segments)
            .map(|segment| segment.units)
            .sum();
        let full = match (request.max_total_tokens, request.pool_storage_full_bytes) {
            (Some(tokens), _) => {
                cache.token_units(tokens as u64, request.max_request_pool_size as u64)
            }
            (None, Some(bytes)) => (bytes as u64 / unit_bytes).saturating_sub(reserved_units),
            (None, None) => u64::from(now) + u64::from(inherited),
        };
        let agreed = group.all_reduce_min(&[full as i64])?;
        let full = u32::try_from(agreed[0].max(0))
            .map_err(|_| Error::Invalid("KV pool exceeds the unit index range".into()))?;
        let head = PlaneLayout::whole_granules(
            1 + self.warmup_units + cache.progress_units(),
            cache.plane_bytes,
            granularity,
        );
        let tail = crate::memory::prefix_cache::planes::segmented_units(
            full.saturating_sub(head + inherited),
            cache.plane_bytes,
            granularity,
        );
        let mut layout = PlaneLayout::successor(
            request.world,
            head,
            tail,
            cache.plane_bytes,
            granularity,
            predecessor,
        )?;
        let created = match layout.tail_segment_units().filter(|_| !request.stage_cache) {
            Some(units) => (now.saturating_sub(head) / units) as usize,
            None => 0,
        };
        layout.defer_tail(created);
        tracing::info!(
            head,
            inherited,
            tail,
            created_tail_segments = created,
            "laid out a successor world's KV cache"
        );
        if now < head {
            return Err(Error::Resource(
                "the device holds less than a successor world's head cache segment",
            ));
        }
        Ok(layout)
    }

    pub(crate) fn finish(self, cache: Option<Arc<CachePlan>>) -> Plan {
        Plan {
            info: self.info,
            arena: self.arena,
            fixed_device_bytes: self.fixed,
            latent: self.latent,
            cache,
            planes: self.planes,
            warmup_units: self.warmup_units,
        }
    }
}

/// The layout of a media worker, sized by its request slots.
pub(crate) fn request_layout(
    report: Report,
    request: &CapacityRequest,
    queue_depth: usize,
    media: &MediaFacts,
    host_components: bool,
) -> Result<Plan> {
    let device_exports = report.device_exports;
    let mut info = base_info(report, request)?;
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
    info.video_denoiser = media.video_denoiser.clone();
    if let Some(plan) = &media.latent {
        info.latent_page_units = narrow(plan.page_units, "latent page units")?;
        info.latent_pages = narrow(plan.num_pages, "latent pages")?;
    }
    let arena = ArenaCapacity::for_requests(&RequestBounds {
        depth: queue_depth,
        calls: info.max_batch_calls as usize,
        requests: slots,
        ranks: request.world_size,
        product_bytes_per_request: media.local_product_bytes,
        concurrent_imports: media.artifact_import_regions,
        device_exports,
    })?;
    info.host_task_capacity = if host_components {
        1
    } else {
        narrow(arena.host_task_inflight, "host task capacity")?
    };
    info.transfer_tickets = narrow(arena.transfer_tickets, "transfer tickets")?;
    Ok(Plan {
        info,
        arena,
        fixed_device_bytes: BTreeMap::new(),
        latent: media.stored_latent(),
        cache: None,
        planes: None,
        warmup_units: 0,
    })
}

/// This rank's storage for every candidate slot count from `minimum` to
/// `maximum`: state, retained products, the product store with its export
/// pool (when `device_exports`) and the latent pool of each count.
pub(crate) fn slot_requirements(
    request: &CapacityRequest,
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
            calls: request.max_batch_calls.min(count),
            requests: count,
            ranks: request.world_size,
            product_bytes_per_request: media.local_product_bytes,
            concurrent_imports: 0,
            device_exports,
        })?;
        let latent_bytes = match media.stored_latent() {
            Some(plan) => {
                // Every request slot owns the same run of pages after the
                // padding page.
                let pages = count * ((plan.num_pages - 1) / plan.request_pool_size) + 1;
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
mod tests {
    use super::*;
    use crate::config::Device;
    use crate::memory::{CacheDescription, CachePlan, Checkpoint};
    use crate::parallel::host::HostGroup;
    use platform::DeviceId;

    /// An 8 GiB device, which is enough for a small dense cache and leaves room
    /// for the fixed storage to be visible in the answer.
    const TOTAL: u64 = 8 << 30;

    fn report() -> Report {
        Report {
            endpoint: ipc::WorkerEndpoint {
                host: "rank".to_owned(),
                address_space: "space".to_owned(),
                incarnation: "incarnation".to_owned(),
            },
            device_exports: false,
            fabric_handles: false,
            supported_calls: Vec::new(),
            media_components: Default::default(),
            components: Vec::new(),
        }
    }

    fn request() -> CapacityRequest {
        CapacityRequest {
            device: Device::Cuda(DeviceId(0)),
            block_size: Some(64),
            pool_storage_bytes: None,
            pool_storage_full_bytes: None,
            mem_fraction_static: 0.5,
            max_batch_calls: 16,
            max_batch_tokens: 2048,
            max_total_tokens: Some(4096),
            max_request_pool_size: 8,
            encoder_cache_entries: 0,
            segments: None,
            stage_cache: false,
            predecessor: None,
            world: 0,
            world_size: 1,
            decode_graph_batch_sizes: Vec::new(),
            prefill_graph_token_sizes: Vec::new(),
            flow_graph_shapes: Vec::new(),
        }
    }

    fn facts() -> TokenFacts {
        TokenFacts {
            image_processor: false,
            vision_feature_bytes: 0,
            latent_feature_bytes: 0,
            latent: None,
        }
    }

    fn dense(facts: CacheGeometry) -> impl FnMut(usize) -> Result<Option<Arc<CachePlan>>> {
        move |size| {
            Ok(Some(Arc::new(CachePlan::new(CacheDescription::dense(
                size as u32,
                facts.layers,
                facts.kv_heads,
                facts.head_dim,
                ipc::PrefixCacheDtype::BFloat16,
            ))?)))
        }
    }

    fn geometry() -> CacheGeometry {
        CacheGeometry {
            layers: 2,
            kv_heads: 4,
            head_dim: 64,
        }
    }

    /// The rank facts a planner is given, with the cache geometry as the class states it.
    fn rank_facts(cache: Option<CacheGeometry>) -> RankFacts<'static> {
        RankFacts {
            device: Device::Cuda(DeviceId(0)),
            host: "rank",
            world: 0,
            world_size: 1,
            queue_depth: 2,
            block_size: Some(64),
            max_batch_calls: 16,
            max_batch_tokens: 2048,
            max_request_pool_size: 8,
            max_model_len: 4096,
            mem_fraction_static: 0.5,
            cache,
            kv_dtype: ipc::PrefixCacheDtype::BFloat16,
        }
    }

    /// The two planners are told apart by whether the rank keeps a paged cache, and each refuses
    /// the other's value by name. A zero geometry standing for absence would be fitted rather than
    /// refused, which is the failure this makes unrepresentable: the plan would then book a cache
    /// the class never addresses.
    ///
    /// Both refusals are answered before the device is read, so neither needs one.
    #[test]
    fn a_planner_refuses_the_other_class_explicitly() {
        let group = HostGroup::single(0);
        let absent = rank_facts(None);
        let Err(refused) = plan_rank(&absent, report(), &group) else {
            panic!("a rank that states no geometry must not be planned a cache");
        };
        let said = format!("{refused}");
        assert!(
            said.contains("plan_media"),
            "a rank with no geometry must be pointed at `plan_media`, and the refusal said {said:?}"
        );

        let present = rank_facts(Some(geometry()));
        let Err(refused) = plan_media(&present, report(), &group) else {
            panic!("a rank that keeps a paged cache must not be planned by the media planner");
        };
        let said = format!("{refused}");
        assert!(
            said.contains("plan_rank"),
            "a rank with a geometry must be pointed at `plan_rank`, and the refusal said {said:?}"
        );
    }

    fn plan(geometry: CacheGeometry) -> Plan {
        token_plan(
            &request(),
            report(),
            &facts(),
            1,
            /* width */ 64,
            TOTAL,
            crate::capacity::graph_storage_budget_bytes(TOTAL as i64),
            None,
            &HostGroup::single(0),
            dense(geometry),
        )
        .unwrap_or_else(|error| {
            panic!("token_plan refused a dense plan of an 8 GiB device: {error}")
        })
    }

    /// The pool the plan fits plus its fixed storage is what the plan books,
    /// and it stays inside the fraction of the device the process may hold.
    /// The fixed figure is what the cache pool's own owner charges for its
    /// tables and unit columns, so the two books agree by construction only if
    /// the caller charges them the same way - which is the point of the rule.
    #[test]
    fn the_pool_and_its_fixed_storage_stay_inside_the_grant() {
        let plan = plan(geometry());
        let cache = plan.cache.as_ref().expect("the model's cache is planned");
        let units = u64::from(
            plan.info
                .prefix_cache
                .as_ref()
                .expect("the rank advertises its cache")
                .num_units,
        );
        let fixed = plan.fixed_device_bytes[&Device::Cuda(DeviceId(0))];
        let booked = fixed + units * cache.unit_bytes;
        let grant = crate::capacity::quota(TOTAL, request().mem_fraction_static);
        assert!(
            booked <= grant,
            "the pool books {booked} bytes of a {grant}-byte grant"
        );
    }

    /// A plan's booked regions are the plan's own figures, region by region, and their sum is the
    /// plan's booked total - so a reader that compares a region with the pool that maps it is
    /// comparing like with like.
    ///
    /// A region added to the plan and not to `Region` does not reach this test: `Region::slot` and
    /// `Region::ALL` are exhaustive over the enum and of the length it counts, so an unlisted
    /// variant fails to compile. That is the tie this test exists for - a sum can be short only by
    /// naming a region the plan does not book.
    #[test]
    fn a_booked_plan_names_every_region_once_and_sums_to_its_parts() {
        // The caller's own device total, because these figures are the ones a run
        // reported: a plan that fits there is a plan that fits anywhere this
        // branch serves.
        let device = Device::Cuda(DeviceId(0));
        let reserved = crate::capacity::reserved_bytes(E2E_TOTAL, /* prepares */ true);
        let graph = crate::capacity::graph_storage_bytes(E2E_TOTAL as i64, true, false);
        let plan = fitted(E2E_TOTAL, reserved).expect("the deployment's device fits a dense plan");
        let booked_plan = TokenPlan {
            plan,
            width: 0,
            reserved: graph,
            shadow: 0,
        };
        let snapshot = booked_plan.state_snapshot_bytes();
        let booked_plan = TokenPlan {
            shadow: reserved - graph - snapshot,
            ..booked_plan
        };
        // What a rank maps before the plan runs, stated here because the plan is
        // fitted into the bytes the weights leave.
        let weights = 1 << 30;
        let booked = booked_plan.booked(device, weights);

        let cache = booked_plan
            .plan
            .cache
            .as_ref()
            .expect("the cache is planned");
        let units = u64::from(
            booked_plan
                .plan
                .info
                .prefix_cache
                .as_ref()
                .expect("the rank advertises its cache")
                .num_units,
        );
        let expected = [
            (Region::Weights, weights),
            (Region::Cache, units * cache.unit_bytes),
            (Region::Graph, graph),
            (Region::Shadow, reserved - graph - snapshot),
            (Region::Snapshot, snapshot),
            (Region::Fixed, booked_plan.plan.fixed_device_bytes[&device]),
        ];
        assert_eq!(
            booked.regions().collect::<Vec<_>>(),
            expected.to_vec(),
            "the plan's booked regions are not the plan's own figures"
        );
        assert_eq!(
            booked.total(),
            expected.iter().map(|(_, bytes)| *bytes).sum::<u64>(),
            "the booked total is not the sum of the regions it names"
        );
        // The two regions a pool maps whole reproduce the caller's reservation to
        // the byte: a plan booked the graph range and forgot the shadow region is
        // the failure this is here to catch.
        assert_eq!(
            booked.get(Region::Graph) + booked.get(Region::Shadow) + booked.get(Region::Snapshot),
            reserved,
            "the graph range and the scratch region do not add up to the reservation"
        );
        assert_eq!(Region::ALL.len(), Region::COUNT);
        let mut named = Region::ALL.to_vec();
        named.sort_unstable();
        named.dedup();
        assert_eq!(
            named.len(),
            Region::COUNT,
            "a region is named twice in `Region::ALL`"
        );
    }

    /// The device total the caller's runs planned into, read from the run's own line: the graph
    /// reservation it logged is two pools of the device's tenth
    /// (`graph_reservation=39579549696`), so the device is five times that.
    const E2E_TOTAL: u64 = 5 * 39_579_549_696;

    /// One plan of `total_bytes` with `reserved` bytes set aside before any unit.
    fn fitted(total_bytes: u64, reserved: u64) -> Result<Plan> {
        token_plan(
            &CapacityRequest {
                max_total_tokens: None,
                mem_fraction_static: 0.7,
                ..request()
            },
            report(),
            &facts(),
            1,
            /* width */ 64,
            total_bytes,
            reserved,
            None,
            &HostGroup::single(0),
            dense(geometry()),
        )
    }

    /// A grant that holds the graph range but not the scratch beside it is refused, and the same
    /// grant with the shadow region added admits the cache.
    ///
    /// The figures are the caller's deployment, not a made-up device: `E2E_TOTAL` is the device the
    /// runs plan into, the graph range is the two pools the run logged, and `mem_fraction_static`
    /// is 0.7 as the run's own placement stated.
    ///
    /// What this case models is a plan whose grant binds - no token capacity, so the pool is sized
    /// by the storage left rather than by the model's context. The served deployment is not that
    /// shape: `plan_rank` states the model's context length, so its cache is demand-bound at about
    /// 4 GB and the device keeps tens of gigabytes of slack. This case exists for the shape where
    /// the grant does bind, and for the invariant that a plan reserves what the pools map before it
    /// fits a unit - not as a reproduction of the deployment's plan.
    #[test]
    fn a_grant_that_holds_the_graph_range_without_the_scratch_is_refused() {
        let graph = crate::capacity::graph_storage_bytes(E2E_TOTAL as i64, true, false);
        assert_eq!(
            graph, 39_579_549_696,
            "the graph range of the deployment the runs plan into"
        );
        let shadow = crate::capacity::graph_scratch_bytes(E2E_TOTAL as i64);
        assert_eq!(shadow, 2_473_721_856, "the shadow region of that device");
        // What a rank reserves before it fits a unit, from the one function that answers it: the
        // graph range and the shadow together. A reservation that lost the shadow is the defect
        // this case exists for, and it fails here before the admission cases below.
        let reserved = crate::capacity::reserved_bytes(E2E_TOTAL, true);
        assert_eq!(
            reserved,
            graph + shadow,
            "the storage a rank reserves on this device: the graph range and the shadow region"
        );
        // A generous grant, only to learn the plan's own fixed storage and unit size.
        let planned = fitted(E2E_TOTAL, reserved)
            .expect("the deployment's device holds its own graph range and scratch");
        let fixed = planned.fixed_device_bytes[&Device::Cuda(DeviceId(0))];
        let unit = planned
            .cache
            .as_ref()
            .expect("the model's cache is planned")
            .unit_bytes;
        // The least a cache serves is the sentinel unit and one page of its one group; the two
        // cases below give that twice, so a rounding in the reservation cannot decide the answer.
        let least = 2 * unit;
        let with = fixed + reserved + 2 * least;
        let without = with - shadow;
        assert!(
            fitted(without, reserved).is_err(),
            "a grant of {without} bytes holds the graph range and the fixed storage, and not the \
             {shadow}-byte shadow region: it must be refused, not fitted"
        );
        let admitted =
            fitted(with, reserved).expect("the same grant with the shadow region holds the cache");
        let units = admitted
            .info
            .prefix_cache
            .as_ref()
            .expect("the rank advertises its cache")
            .num_units;
        assert!(
            units >= 2,
            "the admitted plan serves a cache: {units} units"
        );
    }

    /// A model that keeps a state per unit pays for its snapshot out of the same grant, so the
    /// pool it fits is smaller by the snapshot's share - and what the rank maps afterwards (the
    /// units, the fixed storage, the graph range, the shadow region and the snapshot) stays inside
    /// the grant it planned into.
    ///
    /// The snapshot is the figure design 4.5 copies a unit's checkpoint through, and it is
    /// reserved once beside the shadow region: a plan that fitted its units into the bytes the
    /// scratch will take maps past the device after the model is loaded, where the refusal costs
    /// the whole startup.
    #[test]
    fn a_state_snapshot_is_reserved_out_of_the_grant_before_the_units() {
        // No token capacity and no stated pool storage: the fit is bounded by the grant alone, so
        // a byte reserved for the scratch is a byte no unit is fitted into.
        let request = CapacityRequest {
            max_total_tokens: None,
            ..request()
        };
        let reserved = crate::capacity::graph_storage_budget_bytes(TOTAL as i64);
        let mut state = CacheDescription::dense(64, 2, 4, 64, ipc::PrefixCacheDtype::BFloat16);
        state.groups[0].state = true;
        state.checkpoints = vec![Checkpoint {
            first_plane: 0,
            planes: 8,
            fields: Vec::new(),
        }];
        let snapshot = CachePlan::new(state.clone())
            .expect("a state group with its checkpoints is a plan")
            .state_snapshot_bytes();
        let unit_bytes = CachePlan::new(state.clone())
            .expect("a state group with its checkpoints is a plan")
            .unit_bytes;
        assert!(
            snapshot > unit_bytes,
            "the case is only falsifying when the snapshot costs more than one unit: {snapshot} \
             bytes of snapshot against a {unit_bytes}-byte unit"
        );
        let planned = token_plan(
            &request,
            report(),
            &facts(),
            1,
            /* width */ 64,
            TOTAL,
            reserved,
            None,
            &HostGroup::single(0),
            |_size| Ok(Some(Arc::new(CachePlan::new(state.clone())?))),
        )
        .expect("an 8 GiB device plans a state-group cache");
        let cache = planned
            .cache
            .as_ref()
            .expect("the model's cache is planned");
        let units = u64::from(
            planned
                .info
                .prefix_cache
                .as_ref()
                .expect("the rank advertises its cache")
                .num_units,
        );
        let fixed = planned.fixed_device_bytes[&Device::Cuda(DeviceId(0))];
        let mapped = units * cache.unit_bytes + fixed + reserved + snapshot;
        assert!(
            mapped <= TOTAL,
            "the rank maps {mapped} bytes into a {TOTAL}-byte grant: {units} units of {} bytes, \
             {fixed} bytes of fixed storage, {reserved} bytes reserved for the graph range and the \
             shadow and {snapshot} bytes of state snapshot",
            cache.unit_bytes
        );
    }

    /// Every unit a layer's planes are made of is charged: a deeper model holds
    /// fewer units in the same grant and a larger fixed figure.
    ///
    /// The grant decides the pool here, so the request states no token demand:
    /// a plan that states one is fitted to the demand and both models fit the
    /// same units of it, which is a different question.
    #[test]
    fn a_deeper_model_holds_fewer_units_of_a_larger_unit() {
        let plan = |layers: u32| {
            token_plan(
                &CapacityRequest {
                    max_total_tokens: None,
                    ..request()
                },
                report(),
                &facts(),
                1,
                /* width */ 64,
                TOTAL,
                crate::capacity::graph_storage_budget_bytes(TOTAL as i64),
                None,
                &HostGroup::single(0),
                dense(CacheGeometry {
                    layers,
                    ..geometry()
                }),
            )
            .expect("the device has room for this cache")
        };
        let shallow = plan(2);
        let deep = plan(16);
        let units = |plan: &Plan| plan.info.prefix_cache.as_ref().unwrap().num_units;
        assert!(units(&deep) < units(&shallow));
        assert!(
            deep.cache.as_ref().unwrap().unit_bytes > shallow.cache.as_ref().unwrap().unit_bytes
        );
    }

    /// Every stream column is charged once per unit: a rank that admits more
    /// imports fits fewer units of the same grant, and its fixed figure does not
    /// grow by the same bytes - the columns follow the unit count, not the
    /// tables.
    ///
    /// Grant-bound again: a stated token demand caps both ranks at the demand,
    /// so the column charge would be invisible in the answer.
    #[test]
    fn more_import_workers_hold_fewer_units_of_the_same_grant() {
        let plan = |calls: usize| {
            let mut request = CapacityRequest {
                max_total_tokens: None,
                ..request()
            };
            request.max_batch_calls = calls;
            token_plan(
                &request,
                report(),
                &facts(),
                1,
                64,
                TOTAL,
                crate::capacity::graph_storage_budget_bytes(TOTAL as i64),
                None,
                &HostGroup::single(0),
                dense(geometry()),
            )
            .expect("the device has room for this cache")
        };
        let narrow = plan(1);
        let wide = plan(16);
        let units = |plan: &Plan| plan.info.prefix_cache.as_ref().unwrap().num_units;
        assert!(
            units(&wide) < units(&narrow),
            "a rank admitting 16 calls fits {} units, one admitting 1 fits {}: the import \
             workers' columns are not charged per unit",
            units(&wide),
            units(&narrow)
        );
        // The fixed figure moves too - an import worker's own scratch is fixed - so what the
        // units' difference proves is that the columns are charged per unit: were they charged
        // with the tables, both ranks would fit the same units.
    }
}

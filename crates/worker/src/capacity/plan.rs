//! Startup memory planning: request slots, cache units and the
//! [`MainProcessInfo`] the rank reports.
//!
//! Planning runs after the model is loaded, so measured device storage
//! includes the weights. Every quantity the engine schedules against must be
//! identical on the ranks of a group, so the cache unit count is agreed over
//! the group's [`HostGroup`] with a minimum reduction.
//!
//! A rank that keeps a paged cache places its pool beside fixed storage: the
//! product store's relay arenas and the storage the prefix cache reserves
//! (block tables, request-slot banks, importer scratch). A pool sized from
//! storage takes the grant left after those, the graph range and the scratch
//! region. A media rank keeps no paged cache: every resident request keeps its
//! whole media state, so nothing is fitted.

use std::collections::BTreeMap;

use std::sync::Arc;

use ipc::{CallKind, ComponentInfo, MainProcessEndpoint, MainProcessInfo, MediaCall};
use proto::config::EngineConfig;

use super::{ArenaCapacity, CacheCapacity, CacheDemand, TokenBounds, call_window};
use crate::config::Device;
use crate::memory::CachePlan;
use crate::memory::prefix_cache::PrefixCache;
use crate::memory::prefix_cache::planes::{PlaneLayout, UnitColumn};
use crate::parallel::host::HostGroup;
use crate::{Error, Result};

/// Unit columns the computations over the prefix cache bind beside the import
/// workers': one, on the device's primary stream.
const COMPUTATION_COLUMNS: usize = 1;

/// What a rank states about itself to the memory planner, beside the
/// deployment's [`EngineConfig`]: where it runs, what its captured catalog
/// admits, the cache its model reads and the planes it adopts.
///
/// Every figure is one the engine process owns or has already agreed; nothing
/// is inferred, because a wrong admission bound is a silently smaller cache
/// rather than an error.
pub struct MemoryRequest {
    /// The device this rank serves.
    pub device: Device,
    /// The epoch whose cache planes the rank lays out: its own, or the epoch a
    /// standby adopts the planes of.
    pub epoch: u64,
    /// Requests the scheduler may have in flight, which the daemon places.
    pub queue_depth: usize,
    /// Most calls one batch may carry: the largest bucket of the rank's
    /// catalog, which is its admission bound because a larger batch is refused
    /// before a kernel runs.
    pub max_batch_calls: usize,
    /// Most tokens one batch may carry, bounded the same way.
    pub max_batch_tokens: usize,
    /// The paged cache the rank's model reads, at the model's own page size, or
    /// `None` for a rank that keeps no paged cache - a media rank, whose
    /// resident state is each request's own whole footprint rather than pages
    /// a scheduler addresses.
    pub cache: Option<Arc<CachePlan>>,
    /// The planes a standby adopts: the cache of the epoch it stands by for,
    /// which it maps and did not size. `None` for a rank that lays its planes
    /// out itself.
    pub segments: Option<ipc::CacheTable>,
    /// Device bytes the rank's expert transport takes at its first capture,
    /// after the cache is fitted, or zero for a rank that exchanges no experts.
    pub transport_bytes: u64,
}

/// Model-independent values of the report.
pub struct Report {
    pub endpoint: MainProcessEndpoint,
    /// The rank publishes device products over CUDA VMM, which keeps an
    /// export pool on its primary device.
    pub device_exports: bool,
    pub fabric_handles: bool,
    pub supported_calls: Vec<CallKind>,
    pub media_components: BTreeMap<MediaCall, String>,
    pub components: Vec<ComponentInfo>,
}

/// The resolved memory of a rank: what the main process builds its resident
/// memory from, with the figures no owner states - the widest block table a
/// request needs, which follows from the cache's page size, and the two
/// reservations the rank takes out of its grant before a cache unit is fitted
/// (the graph range and the shadow region beside it), which the fit takes as an
/// input and would otherwise not be readable back.
pub struct MemoryConfig {
    /// The capacity the rank reports to the engine.
    pub info: MainProcessInfo,
    /// The host and device arenas of its calls: read where a rank's arenas are
    /// built, which is inside this crate.
    pub(crate) arena: ArenaCapacity,
    /// Fixed bytes by device, beside the cache pool.
    pub fixed_device_bytes: BTreeMap<Device, u64>,
    /// The cache layout, with the agreed unit count in `info.prefix_cache`.
    pub cache: Option<Arc<CachePlan>>,
    /// The segments a device pool's planes are made of; `None` for planes
    /// that do not split.
    pub planes: Option<PlaneLayout>,
    /// The widest block table a request of the model's length needs, or zero
    /// for a rank that keeps no paged cache.
    pub width: u32,
    /// The graph range the rank reserves: one pool per generation a preparation
    /// may hold, mapped whole at the first capture.
    pub reserved: u64,
    /// The shadow region a replaced operator runs in, reserved beside it.
    pub shadow: u64,
}

fn narrow(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::Invalid(format!("{name} exceeds the protocol range")))
}

fn base_info(report: Report, request: &MemoryRequest, slots: usize) -> Result<MainProcessInfo> {
    Ok(MainProcessInfo {
        media_components: report.media_components,
        video_denoiser: None,
        endpoint: report.endpoint,
        fabric_handles: report.fabric_handles,
        components: report.components,
        supported_calls: report.supported_calls,
        max_batch_calls: narrow(request.max_batch_calls, "max batch calls")?,
        max_batch_tokens: narrow(request.max_batch_tokens, "max batch tokens")?,
        request_slots: narrow(slots, "request slots")?,
        prefix_cache: None,
        latent_page_units: 0,
        latent_pages: 0,
        // A video denoising rank maps its host latent pool once its ledger
        // is bound.
        latent_pool: None,
        encoder_pool: None,
        // An image encoding rank states its encoders once its worker
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

impl MemoryConfig {
    /// The device storage this plan books of `device`, region by region.
    ///
    /// `weights` is the checkpoint's mapped bytes, which the plan cannot know:
    /// the store maps them before the plan measures the device, and its figure is
    /// what the fit's grant already excludes.
    pub fn booked(&self, device: Device, weights: u64) -> Booked {
        let cache = self
            .cache
            .as_ref()
            .map(|cache| {
                let units = self
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
        bytes[Region::Fixed.slot()] = self.fixed_device_bytes.get(&device).copied().unwrap_or(0);
        Booked(bytes)
    }

    /// Bytes one unit's state snapshot occupies, which the scratch pool has to hold beside the
    /// shadow region (design 4.5): the cache plan's own figure, and zero when the model keeps no
    /// state per unit.
    pub fn state_snapshot_bytes(&self) -> u64 {
        self.cache
            .as_ref()
            .map_or(0, |cache| cache.state_snapshot_bytes())
    }

    /// The tokens one page of the served cache holds.
    pub fn block_size(&self) -> u32 {
        self.cache
            .as_ref()
            .and_then(|cache| cache.groups.first().map(|group| group.block_size))
            .unwrap_or(0)
    }
}

/// What a rank's process maps after its plan and beside its pools: the modules a precopy loads and
/// the workspaces of the native libraries they call. A rank measured 2.26 GB of it serving
/// Qwen3-30B-A3B over twelve graphs; a cache sized from storage leaves it this room, because the
/// ledger refuses a rank whose process does not fit beside its cache.
const PROCESS_ALLOWANCE: u64 = 4 << 30;

/// Resolve a rank's memory from what it states and the deployment's configuration: the arenas
/// sized and, for a rank that keeps a paged cache, the pool fitted to the device's grant and its
/// unit count agreed over `group`.
///
/// What the rank may plan into is measured here. A cache sized from storage takes what the
/// deployment's quota (`mem_fraction_static` of the device) leaves: the quota less what is mapped by
/// now, which is the weights and the process, and less what the process has yet to map. Nothing
/// else bounds it, so the fraction binds here or nowhere.
pub fn plan_rank(
    config: &EngineConfig,
    request: &MemoryRequest,
    report: Report,
    group: &HostGroup,
) -> Result<MemoryConfig> {
    let (device_total, grant, granularity) = match request.device {
        Device::Cuda(index) => {
            let storage = index.memory_info()?;
            let held = storage.total.saturating_sub(storage.free);
            let quota = crate::capacity::quota(storage.total, config.mem_fraction_static);
            let grant = quota
                .saturating_sub(held.saturating_add(PROCESS_ALLOWANCE))
                .min(storage.free);
            // A device's cache is split into segments of whole allocation granules, which is
            // what lets it give storage back while every address stays valid (7.6).
            let granularity = platform::HandleType::Descriptor.granularity(index)? as u64;
            (storage.total, Some(grant), Some(granularity))
        }
        Device::Cpu => (0, None, None),
    };
    fit(
        config,
        request,
        report,
        device_total,
        grant,
        granularity,
        group,
    )
}

/// The planner over measured figures: `device_total` is the device's size, from which the graph
/// range and the scratch region are reserved; `grant` is the storage a cache sized from storage
/// may take, `None` where the device states none; `granularity` is the device's allocation
/// granularity, in whose granules a device pool splits its planes, `None` on CPU.
fn fit(
    config: &EngineConfig,
    request: &MemoryRequest,
    report: Report,
    device_total: u64,
    grant: Option<u64>,
    granularity: Option<u64>,
    group: &HostGroup,
) -> Result<MemoryConfig> {
    let slots = config.max_running_requests.get();
    let device_exports = report.device_exports;
    let mut info = base_info(report, request, slots)?;
    info.max_unresolved_calls = narrow(
        call_window(request.queue_depth, request.max_batch_calls)?,
        "unresolved calls",
    )?;
    // The product store's relay arenas and the export pool live on the primary device, beside
    // every binding.
    let arena = ArenaCapacity::for_tokens(&TokenBounds {
        depth: request.queue_depth,
        calls: request.max_batch_calls,
        requests: slots,
        device_exports,
    })?;
    info.host_task_capacity = narrow(arena.host_task_inflight, "host task capacity")?;
    info.transfer_tickets = narrow(arena.transfer_tickets, "transfer tickets")?;
    // The graph range and the scratch region are one reservation (4.1, 4.5): the graph pools the
    // rank captures into and the shadow a replaced operator runs in, both reserved at spawn and
    // both out of the grant the cache is fitted into. `reserved_bytes` answers them as one figure
    // so a caller cannot pass half of it - a plan fitted into the bytes the shadow will take leaves
    // the rank to map past the device at its first shadow, after the model load and after every
    // module installed. The two halves are also kept apart, because a reader of the finished plan
    // has to be able to compare each region with the pool that maps it: the graph range is the
    // graph pool's whole reservation, and the scratch region is the shadow plus, for a model with a
    // state group, the snapshot.
    let reserved = crate::capacity::reserved_bytes(device_total, true);
    let graph = crate::capacity::graph_storage_bytes(device_total as i64, true, false);
    let mut memory = MemoryConfig {
        info,
        arena,
        fixed_device_bytes: BTreeMap::from([(request.device, arena.deferred_device_bytes())]),
        cache: None,
        planes: None,
        width: 0,
        reserved: graph,
        shadow: reserved - graph,
    };
    // A rank that keeps no paged cache fits nothing: its plan states an absent cache rather than a
    // cache of zero units, which is what a scheduler reads to know this rank takes no paged
    // request.
    let Some(cache) = &request.cache else {
        return Ok(memory);
    };
    // The scratch region holds the shadow beside the snapshot, so the shadow is what the
    // reservation has left once the snapshot is taken out of it.
    memory.shadow -= cache.state_snapshot_bytes();
    let block_size = cache
        .groups
        .first()
        .map(|group| group.block_size)
        .filter(|size| *size > 0)
        .ok_or_else(|| Error::Invalid("the model's cache states no page size".into()))?;
    memory.width = config.model.max_model_len.div_ceil(block_size);
    // The cache's fixed storage beside its units: block tables of the width a request of the
    // model's length needs, import scratch and the request-slot banks. Every unit index column is
    // charged with each unit instead, because the unit count is not known until the pool is fitted
    // - so no column may be charged here as well.
    let fixed = PrefixCache::fixed_bytes(
        cache,
        0,
        memory.info.request_slots,
        memory.width,
        memory.info.max_unresolved_calls as usize,
        0,
    )?;
    *memory.fixed_device_bytes.entry(request.device).or_default() += fixed;
    // The transport is created at the first capture, when a program exists, so the cache is
    // fitted around what it will take.
    memory.fit_cache(
        cache,
        config,
        request,
        reserved + request.transport_bytes,
        grant,
        granularity,
        group,
    )?;
    memory.cache = Some(Arc::clone(cache));
    Ok(memory)
}

impl MemoryConfig {
    /// Fit the cache pool beside the fixed storage and agree its unit count.
    ///
    /// The pool takes the grant left after the storage the rank reserved (`reserved_bytes`: the
    /// graph range, the shadow region and the expert transport, and this plan's own state
    /// snapshot, charged here because the scratch holds one beside the other) and the fixed
    /// storage, in whole units. Every rank reserves that storage, whatever its graph policy: graph
    /// capture grows into it, and so do the eager allocations warmup makes (operator workspaces
    /// such as an expert kernel's), which a rank that serves every call eagerly makes too. Graph
    /// padding takes no unit: padding rows read the sentinel unit and write no K/V. A configured
    /// token capacity (`kv_cache_config.max_tokens`) sizes the pool instead.
    ///
    /// Every unit index column holds one index per unit and is charged with each unit: the import
    /// workers' columns and the streams the computations over the cache bind. Charging them here
    /// and not in the fixed figure is what keeps a unit's bytes the same whether the pool is fitted
    /// before or after its size is known.
    #[expect(
        clippy::too_many_arguments,
        reason = "the cache, the configuration, the request, the device's figures and the agreement"
    )]
    fn fit_cache(
        &mut self,
        cache: &CachePlan,
        config: &EngineConfig,
        request: &MemoryRequest,
        reserved_bytes: u64,
        grant: Option<u64>,
        granularity: Option<u64>,
        group: &HostGroup,
    ) -> Result<()> {
        let workers = PrefixCache::import_workers(self.info.max_unresolved_calls as usize) as u64;
        let unit_bytes =
            cache.unit_bytes + (workers + COMPUTATION_COLUMNS as u64) * UnitColumn::bytes(1);
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
                .fixed_device_bytes
                .get(&request.device)
                .copied()
                .unwrap_or(0)
                .div_ceil(unit_bytes);

        let tokens = config.kv_cache.max_tokens.map(|tokens| tokens.get());
        if request.device.is_cuda() && tokens.is_none() && grant.is_none() {
            return Err(Error::Invalid(
                "automatic CUDA KV sizing requires a host storage grant".into(),
            ));
        }
        // A standby rank group's worker adopts the segments of the worker it stands by for.
        let adopted = request.segments.as_ref().map(|table| {
            table
                .segments
                .iter()
                .map(|segment| u64::from(segment.units))
                .sum::<u64>()
        });
        let slots = u64::from(self.info.request_slots);
        let mut capacity = CacheCapacity::derive(&CacheDemand {
            pages: &pages,
            token_units: tokens.map(|tokens| cache.token_units(tokens, slots)),
            units: adopted,
            unit_bytes,
            available_bytes: grant,
            reserved_units,
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
            planes = match &request.segments {
                Some(table) => Some(PlaneLayout::adopted(table)),
                None => {
                    // A cache fitted to a capacity rounds up to whole segments: it holds the tokens
                    // it was asked for, where the storage has room. One fitted to the storage
                    // rounds down into it.
                    let above = crate::memory::prefix_cache::planes::segmented_units_above(
                        own,
                        cache.plane_bytes,
                        granularity,
                    );
                    let fits = grant.is_none_or(|bytes| {
                        (u64::from(above) + reserved_units).saturating_mul(unit_bytes) <= bytes
                    });
                    let rounded = if fits {
                        above
                    } else {
                        crate::memory::prefix_cache::planes::segmented_units(
                            own,
                            cache.plane_bytes,
                            granularity,
                        )
                    };
                    tracing::info!(
                        fitted = own,
                        units = rounded,
                        "rounded the KV cache to whole segments"
                    );
                    capacity.num_units = u64::from(rounded);
                    PlaneLayout::own(request.epoch, rounded, cache.plane_bytes, granularity)
                }
            };
            if let Some(layout) = &planes {
                capacity.num_units = u64::from(layout.units());
            }
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
        self.arena.transfer_bytes =
            (capacity.num_units * cache.unit_bytes).max(1) * self.arena.transfer_tickets as u64;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::num::{NonZeroU64, NonZeroUsize};

    use super::*;
    use crate::config::Device;
    use crate::memory::{CachePlan, Checkpoint};
    use crate::parallel::host::HostGroup;
    use platform::DeviceId;

    /// An 8 GiB device, which is enough for a small dense cache and leaves room
    /// for the fixed storage to be visible in the answer.
    const TOTAL: u64 = 8 << 30;

    fn report() -> Report {
        Report {
            endpoint: ipc::MainProcessEndpoint {
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

    /// The deployment's configuration with the four values the planner reads: eight request
    /// slots, a 4096-token context, half the device, and `max_tokens` as the configured token
    /// capacity of the cache (`None` sizes it from the grant).
    fn config(max_tokens: Option<u64>) -> EngineConfig {
        EngineConfig {
            executable: "test".to_owned(),
            model: proto::config::ModelConfig {
                model: "model".to_owned(),
                max_model_len: 4096,
                dtype: proto::request::ModelDtype::BFloat16,
                quantization: Default::default(),
                generation_limits: Default::default(),
                bos: 0,
                eos: Vec::new(),
                end_of_image: 0,
                canvas_sampling: None,
                prefill_outputs: true,
            },
            parallel: proto::config::ParallelConfig {
                tensor_parallel_size: NonZeroUsize::MIN,
                data_parallel_size: NonZeroUsize::MIN,
                engine_host: "localhost".to_owned(),
                worker_groups: Vec::new(),
            },
            deployment: String::new(),
            max_running_requests: NonZeroUsize::new(8).expect("eight slots"),
            max_num_batched_tokens: None,
            attention_backend: String::new(),
            prefill_attention_backend: None,
            decode_attention_backend: None,
            moe_runner_backend: None,
            device: Default::default(),
            scheduler: Default::default(),
            media: Default::default(),
            mem_fraction_static: 0.5,
            load: Default::default(),
            kv_cache: proto::config::KvCacheConfig {
                max_tokens: max_tokens.map(|tokens| NonZeroU64::new(tokens).expect("a capacity")),
                ..Default::default()
            },
            moe: Default::default(),
            cuda_graph: Default::default(),
            worker: Default::default(),
            speculative: Default::default(),
            video: Default::default(),
            profiling: Default::default(),
        }
    }

    /// A dense cache of 64-token pages over `layers` layers of four 64-wide heads.
    fn dense(layers: u32) -> Arc<CachePlan> {
        Arc::new(
            (CachePlan::dense(64, layers, 4, 64, ipc::PrefixCacheDtype::BFloat16))
                .validated()
                .expect("a dense cache is a plan"),
        )
    }

    /// What a rank states, with the cache its model reads.
    fn request(cache: Option<Arc<CachePlan>>) -> MemoryRequest {
        MemoryRequest {
            device: Device::Cuda(DeviceId(0)),
            epoch: 0,
            queue_depth: 1,
            max_batch_calls: 16,
            max_batch_tokens: 2048,
            cache,
            segments: None,
            transport_bytes: 0,
        }
    }

    /// The grant of `TOTAL` at the configuration's fraction.
    fn grant() -> u64 {
        crate::capacity::quota(TOTAL, config(None).mem_fraction_static)
    }

    /// A rank that keeps no paged cache plans none: it states an absent cache rather than a cache
    /// of no units, and no block table width, and still reserves the graph range and the shadow.
    #[test]
    fn a_rank_that_keeps_no_paged_cache_fits_none() {
        let planned = fit(
            &config(Some(4096)),
            &request(None),
            report(),
            TOTAL,
            Some(grant()),
            None,
            &HostGroup::single(0),
        )
        .unwrap_or_else(|error| {
            panic!("a rank that keeps no cache must plan without one: {error}")
        });
        assert!(
            planned.cache.is_none() && planned.info.prefix_cache.is_none(),
            "a cacheless rank states no cache rather than a cache of no units"
        );
        assert_eq!(planned.width, 0, "a cacheless rank addresses no pages");
        assert_eq!(
            planned.reserved + planned.shadow,
            crate::capacity::reserved_bytes(TOTAL, true),
            "a cacheless rank still reserves the graph range and the shadow region"
        );
    }

    /// The block table width is the pages of the model's context at the cache's page size, and
    /// the fixed storage the plan books holds tables of that width.
    #[test]
    fn the_block_tables_are_charged_at_the_models_width() {
        let planned = plan(2);
        assert_eq!(
            planned.width,
            4096 / 64,
            "the width of a 4096-token context"
        );
        let tables = PrefixCache::fixed_bytes(
            &dense(2),
            0,
            planned.info.request_slots,
            planned.width,
            planned.info.max_unresolved_calls as usize,
            0,
        )
        .expect("the cache's fixed storage");
        let narrow = PrefixCache::fixed_bytes(
            &dense(2),
            0,
            planned.info.request_slots,
            0,
            planned.info.max_unresolved_calls as usize,
            0,
        )
        .expect("the cache's fixed storage without tables");
        assert!(tables > narrow, "block tables of width 64 take storage");
        assert_eq!(
            planned.fixed_device_bytes[&Device::Cuda(DeviceId(0))],
            planned.arena.deferred_device_bytes() + tables,
            "the plan books the tables at the width it states"
        );
    }

    /// A plan of a dense cache over `layers` layers, sized by the configured token capacity.
    fn plan(layers: u32) -> MemoryConfig {
        fit(
            &config(Some(4096)),
            &request(Some(dense(layers))),
            report(),
            TOTAL,
            Some(grant()),
            None,
            &HostGroup::single(0),
        )
        .unwrap_or_else(|error| {
            panic!("the planner refused a dense plan of an 8 GiB device: {error}")
        })
    }

    /// The pool the plan fits plus its fixed storage is what the plan books,
    /// and it stays inside the fraction of the device the process may hold.
    /// The fixed figure is what the cache pool's own owner charges for its
    /// tables and unit columns, so the two books agree by construction only if
    /// the caller charges them the same way - which is the point of the rule.
    #[test]
    fn the_pool_and_its_fixed_storage_stay_inside_the_grant() {
        let plan = plan(2);
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
        assert!(
            booked <= grant(),
            "the pool books {booked} bytes of a {}-byte grant",
            grant()
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
        let plan = fitted(E2E_TOTAL, E2E_TOTAL).expect("the deployment's device fits a dense plan");
        let snapshot = plan.state_snapshot_bytes();
        assert_eq!(plan.reserved, graph, "the plan reserves the graph range");
        assert_eq!(
            plan.shadow,
            reserved - graph - snapshot,
            "the plan's shadow is the scratch region less the snapshot"
        );
        // What a rank maps before the plan runs, stated here because the plan is
        // fitted into the bytes the weights leave.
        let weights = 1 << 30;
        let booked = plan.booked(device, weights);

        let cache = plan.cache.as_ref().expect("the cache is planned");
        let units = u64::from(
            plan.info
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
            (Region::Fixed, plan.fixed_device_bytes[&device]),
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

    /// One plan of a device of `device_total` bytes whose grant is `grant` bytes, sized from the
    /// grant.
    fn fitted(device_total: u64, grant: u64) -> Result<MemoryConfig> {
        fit(
            &config(None),
            &request(Some(dense(2))),
            report(),
            device_total,
            Some(grant),
            None,
            &HostGroup::single(0),
        )
    }

    /// A grant that holds the graph range but not the scratch beside it is refused, and the same
    /// grant with the shadow region added admits the cache.
    ///
    /// The figures are the caller's deployment, not a made-up device: `E2E_TOTAL` is the device the
    /// runs plan into and the graph range is the two pools the run logged.
    ///
    /// What this case models is a plan whose grant binds - no token capacity, so the pool is sized
    /// by the storage left rather than by the model's context. The served deployment is not that
    /// shape: it states the model's context length, so its cache is demand-bound at about 4 GB and
    /// the device keeps tens of gigabytes of slack. This case exists for the shape where the grant
    /// does bind, and for the invariant that a plan reserves what the pools map before it fits a
    /// unit - not as a reproduction of the deployment's plan.
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
        let planned = fitted(E2E_TOTAL, E2E_TOTAL)
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
            fitted(E2E_TOTAL, without).is_err(),
            "a grant of {without} bytes holds the graph range and the fixed storage, and not the \
             {shadow}-byte shadow region: it must be refused, not fitted"
        );
        let admitted =
            fitted(E2E_TOTAL, with).expect("the same grant with the shadow region holds the cache");
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
        let reserved = crate::capacity::reserved_bytes(TOTAL, true);
        let mut state = CachePlan::dense(64, 2, 4, 64, ipc::PrefixCacheDtype::BFloat16);
        state.groups[0].state = true;
        state.checkpoints = vec![Checkpoint {
            first_plane: 0,
            planes: 8,
            fields: Vec::new(),
        }];
        let state = Arc::new(
            state
                .validated()
                .expect("a state group with its checkpoints is a plan"),
        );
        let snapshot = state.state_snapshot_bytes();
        let unit_bytes = state.unit_bytes;
        assert!(
            snapshot > unit_bytes,
            "the case is only falsifying when the snapshot costs more than one unit: {snapshot} \
             bytes of snapshot against a {unit_bytes}-byte unit"
        );
        // No token capacity: the fit is bounded by the grant alone, so a byte reserved for the
        // scratch is a byte no unit is fitted into.
        let planned = fit(
            &config(None),
            &request(Some(state)),
            report(),
            TOTAL,
            Some(TOTAL),
            None,
            &HostGroup::single(0),
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
    /// The grant decides the pool here, so the configuration states no token
    /// demand: a plan that states one is fitted to the demand and both models
    /// fit the same units of it, which is a different question.
    #[test]
    fn a_deeper_model_holds_fewer_units_of_a_larger_unit() {
        let plan = |layers: u32| {
            fit(
                &config(None),
                &request(Some(dense(layers))),
                report(),
                TOTAL,
                Some(grant()),
                None,
                &HostGroup::single(0),
            )
            .expect("the device has room for this cache")
        };
        let shallow = plan(2);
        let deep = plan(16);
        let units = |plan: &MemoryConfig| plan.info.prefix_cache.as_ref().unwrap().num_units;
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
            let mut request = request(Some(dense(2)));
            request.max_batch_calls = calls;
            fit(
                &config(None),
                &request,
                report(),
                TOTAL,
                Some(grant()),
                None,
                &HostGroup::single(0),
            )
            .expect("the device has room for this cache")
        };
        let narrow = plan(1);
        let wide = plan(16);
        let units = |plan: &MemoryConfig| plan.info.prefix_cache.as_ref().unwrap().num_units;
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

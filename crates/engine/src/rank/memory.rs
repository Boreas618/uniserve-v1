//! A rank's device memory: its cache's layout and capacity, its recurrent state, and the division
//! of the GPU's quota among its storages (`specs/engine-worker-port.md` §1.1, steps 5-6 and
//! 11-16).
//!
//! A rank divides what its GPU's quota leaves it once its weights are mapped, from figures that
//! are stated or measured by then (`worker::memory::division`): the process's own bytes, the
//! graph region and shadow region the runner's profile run states, the storages its request slots
//! and its cache's tables take, and the expert transport of a union member. The cache's planes
//! take what remains, in whole units agreed over the rank's host group and rounded to whole
//! segments: the memory of the buffer pool's region is the segments the ranks map or create once
//! the session is open (`specs/buffer-pool.md` §14.1).

use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use ipc::batch::Variant;
use nn::family::{StateField, TextModel};
use platform::DeviceId;
use storage::runtime::Segments;
use uniserve_tensor::DType;
use worker::buckets::Buckets;
use worker::memory::cache::segments::{self, PlaneSegment};
use worker::memory::cache::{ATTENTION_WORKSPACE_BYTES, CacheLayout, KvCacheConfig};
use worker::memory::client::StorageClient;
use worker::memory::division;
use worker::memory::state::StateConfig;
use worker::parallel::host::HostGroup;
use worker::runners::causal_lm::CausalLmConfig;

use super::ModelRunner;

/// Tokens one page of a causal language model's paged cache holds: the largest page the attention
/// packages read. The model reads it from the cache it is handed (`nn::KvGroup::page_size`), and a
/// block of the buffer pool's KV classes is one page (`specs/buffer-pool.md` §6).
pub(crate) const PAGE_TOKENS: u32 = 64;

/// What a profile run states of a rank's runner, which the division sizes the rank's pools by.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Peaks {
    /// The runner's region of the graph pool: its input tensors and its forwards' peak
    /// (`Graphs::peak`).
    pub(crate) graph: u64,
    /// The runner's shadow region (`Graphs::shadow_peak`), with the scratch pages of its paired
    /// forwards.
    pub(crate) shadow: u64,
    /// The bytes of host values one replay copies (`Graphs::input_peak`).
    pub(crate) inputs: u64,
}

impl ModelRunner {
    /// Request slots the rank's runner and cache hold: the deployment's running requests.
    pub(crate) fn slots(&self) -> Result<u32> {
        u32::try_from(self.engine_config().max_running_requests.get())
            .context("more request slots than the protocol addresses")
    }

    /// The layout of the rank's cache for `model` (step 5): one attention group of the model's
    /// layers, the state group of a model that keeps a recurrent state, the index keys its
    /// attention reads when it states them, the drafter's draft group when the deployment
    /// speculates, and converted loads of peers' dense pages into an FP8 cache.
    ///
    /// # Errors
    ///
    /// A cache the model's shape cannot lay out, or a cache element type the model's activations
    /// have none of.
    pub(crate) fn cache_layout(&self, model: &dyn TextModel) -> Result<CacheLayout> {
        let shape = model.cache();
        let kv = &self.engine_config().kv_cache;
        let dtype = cache_dtype(model.dtype(), kv.dtype)?;

        // A model that keeps a recurrent state per request slot adds the state group its
        // checkpoints fill, laid out over every run of its recurrent layers.
        let runs = model.state_layers();
        let layout = if runs.is_empty() {
            CacheLayout::dense(
                PAGE_TOKENS,
                shape.layers,
                shape.kv_heads,
                shape.head_dim,
                dtype,
            )
        } else {
            CacheLayout::hybrid(
                PAGE_TOKENS,
                shape.layers,
                shape.kv_heads,
                shape.head_dim,
                dtype,
                &runs,
            )
            .context("describing the model's KV cache and recurrent state")?
        };

        // A model whose attention reads the tokens an indexer selects keeps their index keys
        // beside its keys and values.
        let layout = match shape.index_keys {
            Some(keys) => layout
                .with_index_keys(keys)
                .context("describing the model's index keys")?,
            None => layout,
        };

        // A deployment that speculates keeps its drafter's draft cache in the same pool.
        let drafts = self.engine_config().speculative.num_draft_tokens;
        let layout = match model.drafter().filter(|_| drafts > 0) {
            Some(drafter) => {
                let shape = drafter.cache();
                layout
                    .with_draft(shape.layers, shape.kv_heads, shape.head_dim)
                    .context("describing the drafter's draft cache")?
            }
            None => layout,
        };

        // A deployment that reads other hosts' host KV cache pages (`kv_cache_config.peers`) into
        // an FP8 pool also loads the pages they computed in the dense compute dtype, by converting
        // them as they load (never the other direction). A pool whose groups have no conversion
        // (a state or draft group, index keys) loads only its own shape.
        let converts = kv.host_tier && kv.prefix_caching && !kv.peers.is_empty();
        let layout = if converts && layout.quantized {
            match layout.clone().with_conversion() {
                Ok(converting) => converting,
                Err(why) => {
                    tracing::info!(%why, "the KV cache loads no converted peer page");
                    layout
                }
            }
        } else {
            layout
        };

        layout.validated().context("checking the KV cache's layout")
    }

    /// The cache of `layout` with `units` units, the padding unit included, made of `segments` on a
    /// device (`None` for planes that are not split): the rank's request slots, and rows of the
    /// pages a request of the model's length holds (step 6).
    ///
    /// # Errors
    ///
    /// More request slots than the protocol addresses.
    pub(crate) fn cache_config(
        &self,
        layout: Arc<CacheLayout>,
        units: u32,
        segments: Option<Vec<PlaneSegment>>,
    ) -> Result<KvCacheConfig> {
        let width = self
            .engine_config()
            .model
            .max_model_len
            .div_ceil(PAGE_TOKENS);

        Ok(KvCacheConfig {
            layout,
            units,
            slots: self.slots()?,
            width,
            segments,
            workspace_bytes: ATTENTION_WORKSPACE_BYTES,
        })
    }

    /// The recurrent state of `model` for a runner over `buckets` (step 6), with its drafter's
    /// when the deployment drafts with one that keeps a state; `None` for a model that keeps
    /// none.
    ///
    /// A request owns `drafts + 1` slots of a stepwise field when the deployment speculates, and
    /// its verifications are then lent verify slots and keep no trail; any other speculating
    /// deployment keeps a trail of the tokens of its largest verification.
    ///
    /// # Errors
    ///
    /// More request slots than the protocol addresses.
    pub(crate) fn state_config(
        &self,
        model: &dyn TextModel,
        buckets: &Buckets,
    ) -> Result<Option<StateConfig>> {
        let shapes = model.state_layers();
        if shapes.is_empty() {
            return Ok(None);
        }

        let drafts = self.engine_config().speculative.num_draft_tokens;
        let stepwise = shapes
            .iter()
            .flat_map(|shape| shape.fields.iter())
            .any(|field: &StateField| field.stepwise);
        let owned = if stepwise && drafts > 0 {
            drafts + 1
        } else {
            1
        };
        let trail = if drafts > 0 && !stepwise {
            buckets
                .iter()
                .filter(|bucket| {
                    matches!(
                        bucket.variant,
                        Variant::VerifyGreedy | Variant::VerifySampled
                    )
                })
                .map(|bucket| bucket.tokens)
                .max()
                .unwrap_or(0)
        } else {
            0
        };

        // A drafter whose layers keep a state of their own keeps one live row a slot of each of
        // its fields beside the model's.
        let drafter = model
            .drafter()
            .filter(|_| drafts > 0)
            .map(|drafter| drafter.state())
            .unwrap_or_default();

        Ok(Some(StateConfig {
            shapes,
            slots: self.slots()?,
            rows: buckets.bounds().0,
            trail,
            owned,
            drafter,
        }))
    }

    /// The bytes the graph pool and the scratch pool of `device` map for a runner of `peaks` over
    /// a cache of `layout` (step 13): its region, its shadow region and the state snapshots of a
    /// model that keeps recurrent state, in whole allocation granules.
    ///
    /// # Errors
    ///
    /// When the device states no allocation granularity.
    pub(crate) fn pools(
        &self,
        device: DeviceId,
        layout: &CacheLayout,
        peaks: Peaks,
    ) -> Result<(u64, u64)> {
        Ok(division::pool_bytes(
            granularity(device)?,
            &[peaks.graph],
            &[peaks.shadow],
            layout.state_snapshot_bytes(),
        ))
    }

    /// The units of the cache planes on `device` (steps 11-12 and 14-15): what the GPU's quota
    /// leaves once the weights are mapped, less the graph and scratch `pools`, the recurrent
    /// state's `resident` bytes, the request slots' `slot_bytes`, the bytes `fixed` the cache
    /// keeps beside its planes and the expert transport of a union member, agreed over `host` as
    /// the smallest count of its ranks, and rounded down to whole segments of whole granules.
    ///
    /// # Errors
    ///
    /// The Storage Server's or the driver's refusal, the region the quota cannot hold, or a
    /// cache too small for one segment of whole granules: the memory of a device KV region is
    /// the segments its ranks create, and planes that do not split are none.
    #[expect(
        clippy::too_many_arguments,
        reason = "the division's figures come from six owners: the client, the host group, the \
                  layout, the pools, the state and the cache's tables"
    )]
    pub(crate) fn divide(
        &self,
        client: &StorageClient,
        device: DeviceId,
        host: &HostGroup,
        model: &dyn TextModel,
        layout: &CacheLayout,
        (graph, scratch): (u64, u64),
        resident: u64,
        slot_bytes: u64,
        fixed: u64,
    ) -> Result<u32> {
        let granularity = granularity(device)?;

        // What the process holds beside its pools and the server's storages. The weight store
        // waits for each copy of the load before it returns, so the figure counts the load's
        // work. A communicator states no bytes NVML attributes to no process
        // (`backend::Group`), so NVML's figure is taken whole.
        let nvml = platform::Nvml::init().context("opening NVML")?;
        let process = division::process_bytes(&nvml, device, 0, 0)?;

        let (quota, _) = client.usage(device)?;
        let (_, kept) = client.scope_bytes(device)?;
        let free = device.memory_info()?.free;
        let unreserved =
            division::unreserved(&quota, client.scope(), client.epoch(), kept, process, free)?;

        let remainder = division::remainder(
            unreserved,
            &[
                ("graph pool", graph),
                ("scratch pool", scratch),
                ("resident pool", resident),
                ("request slots", slot_bytes),
                ("cache tables", fixed),
                ("expert transport", self.transport_bytes(device, model)?),
            ],
            0,
            &[],
        )?;

        let slots = u64::from(self.slots()?);
        let capacity = self
            .engine_config()
            .kv_cache
            .max_tokens
            .map(|tokens| layout.token_units(tokens.get(), slots));
        let units = division::cache_units(
            remainder,
            layout.unit_bytes,
            u64::from(fewest_units(layout)),
            capacity,
            host,
        )?;

        // Whole segments of whole granules, an eighth of the cache each at most: what a resize
        // across layouts, a sleep and a wake give back and create again.
        let units = segments::segmented_units(units, layout.plane_bytes, granularity);
        if segments::segment_layout(units, layout.plane_bytes, granularity).is_none() {
            bail!(
                "a cache of {units} units of {} plane bytes holds no segment of whole \
                 {granularity}-byte granules, and a device KV region's memory is the segments its \
                 ranks create",
                layout.plane_bytes
            );
        }
        Ok(units)
    }
}

/// The segments of a cache of `units` units laid out in `layout` whose planes are allocated in
/// granules of `granularity` bytes (step 16), given the region's segment table `table`
/// (`storage::runtime::Opened::segments`): the segments the table states, which the rank maps
/// where their epoch created them (`segments::of_table`), then the segments `epoch` creates for
/// the units past them; and those created alone, as a segment table states them (the segment,
/// its first unit and its units), which the session's head grants.
///
/// The created segments are the ones `segments::own` splits the whole cache into, from the end
/// of the table on: a region of one layout is split alike by every epoch that names it, so a
/// table that ends inside one of them is not this layout's.
///
/// # Errors
///
/// A table whose segments do not lie back to back from unit 0, that covers more than `units` or
/// ends inside a segment of the split, or a cache that does not split.
pub(crate) fn plane_segments(
    epoch: u64,
    units: u32,
    layout: &CacheLayout,
    granularity: u64,
    table: &[(storage::Segment, u32, u32)],
) -> Result<(Vec<PlaneSegment>, Segments)> {
    let mapped = segments::of_table(table)?;
    let covered = segments::units(&mapped);
    if covered > units {
        bail!("the region's segment table covers {covered} units of a cache of {units}");
    }
    let plane_bytes = layout.plane_bytes;
    let split = segments::own(epoch, units, plane_bytes, granularity).with_context(|| {
        format!(
            "a cache of {units} units of {plane_bytes} plane bytes holds no segment of whole \
             {granularity}-byte granules"
        )
    })?;

    let mut first = 0;
    let mut created = Vec::new();
    let mut stated = Vec::new();
    for part in split {
        if first >= covered {
            let segment = storage::Segment {
                epoch: part.segment.epoch,
                ordinal: part.segment.ordinal,
            };
            stated.push((segment, first, part.segment.units));
            created.push(part);
        } else if first + part.segment.units > covered {
            bail!(
                "the region's segment table ends at unit {covered}, inside segment {} of units \
                 {first}..{} this layout splits the cache into",
                part.segment.ordinal,
                first + part.segment.units
            );
        }
        first += part.segment.units;
    }

    let segments = mapped.into_iter().chain(created).collect();
    Ok((segments, stated))
}

/// The bytes of the scratch pages a runner of `config` writes its paired forwards' keys and
/// values into, over a cache of `layout`: `scratch_pages` pages of one unit for each row of the
/// largest bucket (step 13), which the scratch pool holds beside the runner's shadow region.
pub(crate) fn scratch_bytes(config: &CausalLmConfig, layout: &CacheLayout) -> u64 {
    let rows = config.buckets.bounds().0;
    u64::from(rows) * u64::from(config.scratch_pages) * layout.unit_bytes
}

/// The size and alignment unit of the exportable allocations of `device`, which a pool's mapping
/// and a segment's every part take whole.
pub(crate) fn granularity(device: DeviceId) -> Result<u64> {
    let granularity = platform::DeviceAllocation::granularity(device)
        .with_context(|| format!("reading device {}'s allocation granularity", device.0))?;
    Ok(granularity as u64)
}

/// The fewest units a cache of `layout` serves with: the padding unit and one page of every cache
/// group.
pub(crate) fn fewest_units(layout: &CacheLayout) -> u32 {
    1 + layout
        .groups
        .iter()
        .map(|group| group.units_per_page)
        .sum::<u32>()
}

/// The element type the cache pool stores, from the deployment when it states one.
///
/// An unstated cache dtype is the model's own activation dtype, which is what its layers compute
/// in until the deployment asks for a narrower cache. A model whose activations have no cache
/// element type cannot have its pool laid out, and says so rather than taking the nearest.
pub(super) fn cache_dtype(
    activations: DType,
    stated: Option<proto::request::PrefixCacheDtype>,
) -> Result<proto::request::PrefixCacheDtype> {
    if let Some(dtype) = stated {
        return Ok(dtype);
    }
    match activations {
        DType::BF16 => Ok(proto::request::PrefixCacheDtype::BFloat16),
        DType::F16 => Ok(proto::request::PrefixCacheDtype::Float16),
        DType::F32 => Ok(proto::request::PrefixCacheDtype::Float32),
        DType::F8E4M3 => Ok(proto::request::PrefixCacheDtype::Float8E4m3Fn),
        other => {
            bail!("this build lays out no KV cache in a model whose activations are {other:?}")
        }
    }
}

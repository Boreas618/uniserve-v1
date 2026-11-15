//! A rank's device memory: its cache's layout and capacity, its recurrent state, and the division
//! of the GPU's quota among its storages (`specs/engine-worker-port.md` §1.1, steps 5-6 and
//! 11-16).
//!
//! A rank divides what its GPU's quota leaves it once its weights are mapped, from figures that
//! are stated or measured by then (`crate::memory::division`): the process's own bytes, the
//! graph regions and shadow regions the runners' profile runs state, the expert transport of a
//! union member, on a rank that encodes images its region of encoder outputs and its causal
//! runner's feature buffer, and the storages its request slots and its cache's tables take,
//! within its own share of the GPU (its worker group's `mem_fraction_static`). The request slots
//! are the deployment's running requests, or as many as the share holds beside the rest and the
//! cache's fewest units when fewer fit (`division::request_slots`). The cache's planes take what
//! remains, in whole units agreed over the rank's host group and rounded to whole segments: the
//! memory of the buffer pool's region is the segments the ranks map or create once the session is
//! open (`specs/buffer-pool.md` §14.1).
//!
//! A model that keeps a recurrent state per request keeps its checkpoints in the scheduler's
//! state region beside the KV planes, one checkpoint a unit (`specs/engine-worker-port.md` §1.6):
//! a request slot costs one checkpoint beside its storages, and the state region takes, beyond
//! that floor, as many bytes as the KV planes (`division::state_units`). Its units are numbered,
//! created and given back in whole segments of its own, as the KV region's are, its segments
//! created before the KV region's.
//!
//! The region is numbered for what the quota holds as if the rank's epoch served the GPU alone,
//! and the division counts the process as it is before the warm-up, whose instantiated graphs and
//! library state the warm process adds (`specs/engine-worker-port.md` §1, "The process a
//! division counts"). The start creates, before the warm-up, the whole segments of that numbering
//! the quota holds free now ([`first_units`]): the step reads its cache from storage mapped before
//! the warm-up, which keeps the decode step at its device time (a step that reads planes mapped
//! after the warm-up was measured about 6 % slower on a GB200, for a reason not established:
//! `specs/engine-worker-port.md` §1). The warm-up's growth lies outside the quota meanwhile; once
//! warm, the rank gives back the tail segments that no longer fit beside its warm process,
//! reserves it, and an epoch beside a draining predecessor of another layout grows into that
//! epoch's memory by `Grow` as it drains (`specs/updates.md` §3.7).

use std::sync::Arc;

use crate::batch::Variant;
use crate::memory::cache::segments::{self, PlaneSegment};
use crate::memory::cache::{ATTENTION_WORKSPACE_BYTES, CacheLayout, KvCacheConfig};
use crate::memory::client::{StorageClient, quota_limit};
use crate::memory::division;
use crate::memory::state::StateConfig;
use crate::parallel::HostGroup;
use crate::runner::buckets::Buckets;
use crate::runner::causal_lm::{CausalLmConfig, CausalLmRunner};
use anyhow::{Context as _, Result, bail};
use nn::family::{StateField, TextModel};
use platform::DeviceId;
use proto::config::WorkerGroupConfig;
use storage::runtime::Segments;
use tensor::DType;

use super::ModelRunner;

/// What a profile run states of a rank's runner, which the division sizes the rank's pools by.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Peaks {
    /// The runner's region of the graph pool: its input tensors and its forwards' peak
    /// (`Graphs::peak`).
    pub(crate) graph: u64,
    /// The runner's shadow region (`Graphs::shadow_peak`). The scratch pages of its paired
    /// forwards lie beside it in the scratch pool, in the extents of the device's pools
    /// ([`ModelRunner::pools`]).
    pub(crate) shadow: u64,
    /// The bytes of host values one replay copies (`Graphs::input_peak`).
    pub(crate) inputs: u64,
}

impl ModelRunner {
    /// Request slots the rank's runner and cache hold: the deployment's running requests, or the
    /// fewer the division of the rank's share of its GPU decided ([`ModelRunner::with_slots`]).
    pub(crate) fn slots(&self) -> u32 {
        self.slots
    }

    /// Tokens one page of a causal language model's paged cache holds: the deployment's KV block
    /// size (`proto::config::KvCacheConfig::block_tokens`). The model reads it from the cache it is
    /// handed (`nn::KvGroup::page_size`), and a block of the buffer pool's KV classes is one page
    /// (`specs/buffer-pool.md` §6).
    pub(crate) fn page_tokens(&self) -> u32 {
        self.engine_config().kv_cache.block_tokens()
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
                self.page_tokens(),
                shape.layers,
                shape.kv_heads,
                shape.head_dim,
                dtype,
            )
        } else {
            CacheLayout::hybrid(
                self.page_tokens(),
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
    /// device (`None` for planes that are not split), and the state region's units with their
    /// segments for a layout with a state group (`(0, None)` without): the rank's request slots,
    /// and rows of the pages a request of the model's length holds (step 6).
    pub(crate) fn cache_config(
        &self,
        layout: Arc<CacheLayout>,
        units: u32,
        segments: Option<Vec<PlaneSegment>>,
        states: (u32, Option<Vec<PlaneSegment>>),
    ) -> KvCacheConfig {
        let width = self
            .engine_config()
            .model
            .max_model_len
            .div_ceil(self.page_tokens());

        KvCacheConfig {
            layout,
            units,
            slots: self.slots(),
            width,
            segments,
            workspace_bytes: ATTENTION_WORKSPACE_BYTES,
            states,
        }
    }

    /// The recurrent state of `model` for a runner over `buckets` (step 6), with its drafter's
    /// when the deployment drafts with one that keeps a state; `None` for a model that keeps
    /// none.
    ///
    /// A request owns `drafts + 1` slots of a stepwise field when the deployment speculates, and
    /// its verifications are then lent verify slots and keep no trail; any other speculating
    /// deployment keeps a trail of the tokens of its largest verification.
    pub(crate) fn state_config(
        &self,
        model: &dyn TextModel,
        buckets: &Buckets,
    ) -> Option<StateConfig> {
        let shapes = model.state_layers();
        if shapes.is_empty() {
            return None;
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

        Some(StateConfig {
            shapes,
            slots: self.slots(),
            rows: buckets.bounds().0,
            trail,
            owned,
            drafter,
        })
    }

    /// The bytes the graph pool and the scratch pool of `device` map for runners of `peaks` over
    /// a cache of `layout` (step 13): each runner's region and shadow region (the causal runner's,
    /// the first, beside the scratch page set of `config` over `cache`, and a vision runner's),
    /// and the state snapshot of a model that keeps recurrent state, one state unit, in the whole
    /// extents the device's pools map.
    ///
    /// # Errors
    ///
    /// When the device states no pool granularity.
    pub(crate) fn pools(
        &self,
        device: DeviceId,
        (config, cache): (&CausalLmConfig, &KvCacheConfig),
        peaks: &[Peaks],
    ) -> Result<(u64, u64)> {
        let granularity = pool_granularity(device)?;
        let regions: Vec<u64> = peaks.iter().map(|peaks| peaks.graph).collect();
        let mut shadows: Vec<u64> = peaks.iter().map(|peaks| peaks.shadow).collect();
        if let Some(causal) = shadows.first_mut() {
            *causal += scratch_bytes(config, cache, granularity);
        }

        Ok(division::pool_bytes(
            granularity,
            &regions,
            &shadows,
            cache.layout.state_bytes(),
        ))
    }

    /// The device bytes the storages of `slots` request slots take on this rank, serving `model`
    /// over a cache of `layout`, as a runner of that many slots states them (steps 6-7): each
    /// slot's state and the outputs a speculating runner keeps (`CausalLmRunner::slot_bytes`,
    /// `CausalLmRunner::kept_bytes`), the cache's pooled page tables and the rest it keeps beside
    /// its planes (`KvCacheConfig::fixed_bytes`), and the recurrent state of a model that keeps
    /// one (`StateConfig::bytes`). The checkpoint a slot costs in the state region is not in it
    /// ([`Self::divide_unreserved`]).
    ///
    /// # Errors
    ///
    /// A runner config the model refuses.
    pub(crate) fn slot_storage(
        &self,
        model: &dyn TextModel,
        layout: &Arc<CacheLayout>,
        slots: u32,
    ) -> Result<u64> {
        let runner = self.with_slots(slots);
        let least = fewest_units(layout);
        let config = runner.causal_config(model, layout, least)?;

        let state = CausalLmRunner::slot_bytes(&config, model)?
            + CausalLmRunner::kept_bytes(&config, model)?;
        let tables = runner
            .cache_config(
                Arc::clone(layout),
                least,
                None,
                (fewest_states(layout), None),
            )
            .fixed_bytes();
        let recurrent = match runner.state_config(model, runner.buckets()) {
            Some(state) => state.bytes(layout)?,
            None => 0,
        };
        Ok(state + tables + recurrent)
    }

    /// The request slots, the units of the cache planes on `device` (steps 11-12 and 14-15) with
    /// the units of them the rank's start creates, and the same of the state region of a model
    /// that keeps a recurrent state (zero without): `(slots, (units, first), (states, first))`.
    ///
    /// `regions` are the `(name, bytes)` of the rank's regions that do not grow with the request
    /// slots, made at its start: its graph and scratch pools ([`Self::pools`]) and, on a rank that
    /// encodes images, its region of encoder outputs (which the resident pool holds) and its
    /// causal runner's feature buffer, and the expert transport of a union member
    /// ([`ModelRunner::transport_bytes`]).
    ///
    /// The request slots are the deployment's running requests, or as many as what the GPU's
    /// quota holds now leaves once the weights are mapped and those regions and the cache's
    /// fewest units are taken out, when fewer fit ([`division::request_slots`]), each slot's
    /// storages as [`Self::slot_storage`] states them and, for a model that keeps a recurrent
    /// state, one checkpoint; agreed over `host` as the fewest any of its ranks holds, since the
    /// ranks of a group serve the same requests.
    ///
    /// The state region's units are its floor of one checkpoint a slot and as many bytes again as
    /// the KV planes take, in whole segments of its own (`division::state_units`), numbered and
    /// agreed as the planes' units are; the planes take the rest. The start creates the state
    /// region's segments first, as far as the quota holds them now, and the planes' from what is
    /// left.
    ///
    /// The units are the region's numbering: what the GPU's quota leaves once the weights are
    /// mapped as if the epoch served the GPU alone, less those regions and the storages of the
    /// agreed slots, agreed over `host` as the smallest count of its ranks, and rounded down to
    /// whole segments of whole granules. The first units are the segments the start creates
    /// before the warm-up: every whole segment of the numbering the quota holds free now, beside
    /// the process measured before the warm-up ([`first_units`]), agreed over `host` alike; a
    /// stated capacity is created whole when the quota holds it now. Once warm, the rank gives
    /// back the tail segments its quota no longer holds beside its warm process, and an epoch that
    /// starts beside the epoch it succeeds gains what that epoch holds by `Grow` as it drains
    /// (`specs/updates.md` §3.7).
    ///
    /// # Errors
    ///
    /// The Storage Server's or the driver's refusal, a share that holds no request slot (naming
    /// the bytes one needs), the region the quota cannot hold now, or a cache too small for one
    /// segment of whole granules: the memory of a device KV region is the segments its ranks
    /// create, and planes that do not split are none.
    #[expect(
        clippy::type_complexity,
        reason = "the request slots, and the units and first units of each of the two regions"
    )]
    pub(crate) fn divide(
        &self,
        client: &StorageClient,
        device: DeviceId,
        host: &HostGroup,
        model: &dyn TextModel,
        layout: &Arc<CacheLayout>,
        regions: &[(&'static str, u64)],
    ) -> Result<(u32, (u32, u32), (u32, u32))> {
        let granularity = granularity(device)?;
        let (share, _) = self.fractions()?;
        let unreserved = unreserved(client, device, share)?;

        self.divide_unreserved(unreserved, granularity, host, model, layout, regions)
    }

    /// [`Self::divide`] over the bytes the GPU's quota leaves the rank, `(now, alone)` as
    /// [`unreserved`] states them, for a device of allocation granules of `granularity` bytes:
    /// the fixed `regions` (the graph and scratch pools, the expert transport, and on a rank that
    /// encodes images its encoder outputs and image features) are taken out, then the request
    /// slots', and the cache's planes take the rest.
    ///
    /// # Errors
    ///
    /// As [`Self::divide`], but for the measurements.
    #[expect(
        clippy::type_complexity,
        reason = "the request slots, and the units and first units of each of the two regions"
    )]
    pub(crate) fn divide_unreserved(
        &self,
        (now, alone): (u64, u64),
        granularity: u64,
        host: &HostGroup,
        model: &dyn TextModel,
        layout: &Arc<CacheLayout>,
        regions: &[(&'static str, u64)],
    ) -> Result<(u32, (u32, u32), (u32, u32))> {
        // The pools, the slots and the cache's tables are made at the start, so they fit what
        // the quota holds now; the planes and the state region are numbered for what it holds
        // the epoch alone.
        let least = fewest_units(layout);
        let state_bytes = layout.state_bytes();
        let left = division::remainder(
            now,
            regions,
            u64::from(least),
            &[("cache's fewest units", layout.unit_bytes)],
        )?;
        // A slot of a model that keeps a recurrent state costs one checkpoint beside its storages.
        let slots = division::request_slots(self.slots(), left, |slots| {
            let checkpoints = u64::from(slots) * state_bytes;
            Ok::<u64, anyhow::Error>(self.slot_storage(model, layout, slots)? + checkpoints)
        })?;
        let slots = host
            .all_reduce_min(&[i64::from(slots)])
            .context("agreeing the request slots of the ranks")?[0];
        let slots = u32::try_from(slots).context("a negative count of request slots was agreed")?;

        let storage = self.slot_storage(model, layout, slots)?;
        let regions: Vec<(&str, u64)> = regions
            .iter()
            .copied()
            .chain([("request slots", storage)])
            .collect();
        let remainder = division::remainder(now, &regions, 0, &[])?;
        let whole = division::remainder(alone, &regions, 0, &[])?;

        let capacity = self
            .engine_config()
            .kv_cache
            .max_tokens
            .map(|tokens| layout.token_units(tokens.get()));

        // The state region first: its floor and its ratio of the KV planes' bytes, agreed over the host
        // group; the planes take what it leaves.
        let states = if state_bytes > 0 {
            let segment = segments::segment_units(state_bytes, granularity)
                .context("a checkpoint of no bytes splits into no segment")?;
            let kv_capacity = capacity.map(|units| units.saturating_mul(layout.unit_bytes));
            let ratio = self.engine_config().kv_cache.state_ratio;
            let states =
                division::state_units(whole, state_bytes, slots, (kv_capacity, ratio), segment)?;
            let states = host
                .all_reduce_min(&[i64::from(states)])
                .context("agreeing the units of the ranks' state regions")?[0];
            u32::try_from(states).context("a negative count of state units was agreed")?
        } else {
            0
        };
        let state_total = u64::from(states) * state_bytes;
        let units = division::cache_units(
            whole.saturating_sub(state_total),
            layout.unit_bytes,
            u64::from(least),
            capacity,
            host,
        )?;

        // Whole segments, each the fewest units that fill whole granules of every column's
        // allocation: what a start, a grow and a wake create and a release gives back, so the
        // region numbers what the quota holds less under one granule of each allocation.
        let units = segments::segmented_units(units, layout.plane_bytes, granularity);
        if segments::segment_layout(units, layout.plane_bytes, granularity).is_none() {
            bail!(
                "a cache of {units} units of {} plane bytes holds no segment of whole \
                 {granularity}-byte granules, and a device KV region's memory is the segments its \
                 ranks create",
                layout.plane_bytes
            );
        }

        // Every rank of a region creates the same segments of it before its warm-up: every whole
        // segment the quota holds now beside the process measured before the warm-up, the state
        // region's first. The warm-up then grows the process outside the quota, and once warm
        // the rank gives back the tail segments that no longer fit beside it
        // (`epoch::keep_warm_segments`).
        let first_states = first_states(states, state_bytes, granularity, remainder);
        let first = first_units(
            units,
            layout,
            granularity,
            remainder.saturating_sub(u64::from(first_states) * state_bytes),
        );
        let agreed = host
            .all_reduce_min(&[i64::from(first), i64::from(first_states)])
            .context("agreeing the units the ranks create at their start")?;
        let first = u32::try_from(agreed[0]).context("a negative count of units was agreed")?;
        let first_states =
            u32::try_from(agreed[1]).context("a negative count of state units was agreed")?;
        Ok((slots, (units, first), (states, first_states)))
    }

    /// This rank's share of its GPU's storage, and the fraction of the GPU its epoch's quota of
    /// the GPU is limited to (`WorkerGroupConfig::fraction`, `WorkerGroupConfig::device_fraction`):
    /// its worker group's `mem_fraction_static`, or the deployment's, and the shares of every
    /// rank the deployment places on the GPU.
    ///
    /// # Errors
    ///
    /// A rank the deployment's worker groups do not place.
    pub(crate) fn fractions(&self) -> Result<(f64, f64)> {
        let config = self.engine_config();
        let groups = &config.parallel.worker_groups;
        let rank = self.rank();
        let (group, position) = WorkerGroupConfig::place_rank(groups, rank as usize)
            .with_context(|| format!("the deployment's worker groups place no rank {rank}"))?;
        let group = &groups[group];

        let deployment = config.mem_fraction_static;
        Ok((
            group.fraction(deployment),
            WorkerGroupConfig::device_fraction(groups, &group.ranks[position], deployment),
        ))
    }
}

/// The units of a cache of `units` units laid out in `layout`, split into segments of whole
/// `granularity`-byte granules, that `remainder` bytes hold now: the planes that are not split
/// (flags, scales, index keys) are made for every unit at once, and of the value and state
/// planes the segments from the first on, as many whole ones as the rest holds. Zero when the
/// unsplit planes do not fit, or the cache does not split.
pub(crate) fn first_units(
    units: u32,
    layout: &CacheLayout,
    granularity: u64,
    remainder: u64,
) -> u32 {
    let Some((count, per_segment)) =
        segments::segment_layout(units, layout.plane_bytes, granularity)
    else {
        return 0;
    };

    // One unit of the split planes over every column, which its segment holds; the rest of a
    // unit's bytes is in planes that are not split.
    let split = layout.split_bytes();
    let unsplit = u64::from(units) * layout.unit_bytes.saturating_sub(split);
    let segment = (u64::from(per_segment) * split).max(1);

    let fitting = remainder
        .checked_sub(unsplit)
        .map_or(0, |left| left / segment);
    u32::try_from(fitting).unwrap_or(u32::MAX).min(count) * per_segment
}

/// The units of a state region of `states` units of `state_bytes` bytes, split into segments of
/// whole `granularity`-byte granules, that `remainder` bytes hold now: as many whole segments
/// from the first as it holds. Zero for a region of no unit or no byte.
pub(crate) fn first_states(states: u32, state_bytes: u64, granularity: u64, remainder: u64) -> u32 {
    let Some(per_segment) = segments::segment_units(state_bytes, granularity) else {
        return 0;
    };
    let segment = (u64::from(per_segment) * state_bytes).max(1);
    let fitting = u32::try_from(remainder / segment).unwrap_or(u32::MAX);
    fitting.saturating_mul(per_segment).min(states)
}

/// The bytes of `device`'s quota a starting rank of `client`'s scope may divide (steps 11-12),
/// `(now, alone)`: what the GPU's quota leaves once the process's own bytes, the epoch's
/// reservations and what the scope keeps are taken out; and the same as if the rank's epoch
/// served the GPU alone, the storages other epochs of the deployment hold apart from it being
/// given back (`division::held_elsewhere`). Both are at most the rank's own `share` of the GPU
/// less the storages its scope holds there.
///
/// # Errors
///
/// The Storage Server's, NVML's or the driver's refusal, or a quota that states no limit.
pub(crate) fn unreserved(
    client: &StorageClient,
    device: DeviceId,
    share: f64,
) -> Result<(u64, u64)> {
    // What the process holds beside its pools and the server's storages. The weight store waits
    // for each copy of the load before it returns, so the figure counts the load's work. A
    // communicator states no bytes NVML attributes to no process (`ops::Group`), so NVML's
    // figure is taken whole.
    let nvml = platform::Nvml::init().context("opening NVML")?;
    let process = division::process_bytes(&nvml, device, 0, 0)?;

    let (quota, _) = client.usage(device)?;
    let (held, kept) = client.scope_bytes(device)?;
    let memory = device.memory_info()?;
    let scope = client
        .scope()
        .context("the head's connection holds no rank's quota")?;
    let epoch = client.epoch();

    let now = division::unreserved(&quota, scope, epoch, kept, process, memory.free)?;
    let elsewhere = division::held_elsewhere(&quota, epoch);
    let alone = division::unreserved(
        &quota,
        scope,
        epoch,
        kept.saturating_add(elsewhere),
        process,
        memory.free,
    )?;

    // A rank of a group that states its own share holds no more of the GPU than it, whatever
    // the deployment's quota of the GPU holds for the other ranks placed there. The process's own
    // bytes are not the rank's alone (ranks of one process on one device share its context, and
    // NVML states the process whole), so the quota above counts them, and the share does not.
    let own = quota_limit(memory.total, share).saturating_sub(held);
    tracing::info!(
        device = device.0,
        process,
        held,
        now = now.min(own),
        alone = alone.min(own),
        "the bytes of the GPU's quota the rank divides"
    );
    Ok((now.min(own), alone.min(own)))
}

/// The segments of a region of `units` units whose memory is allocated in granules of
/// `granularity` bytes, a column of a unit taking `part_bytes` (a plane of the KV planes, or a
/// whole state unit) (step 16), given the region's segment table `table`
/// (`storage::runtime::Opened::segments`): the segments the table states, which the rank maps
/// where their epoch created them (`segments::of_table`), then the segments `epoch` creates for
/// the units past them up to `held`, the units the GPU's quota holds now
/// ([`ModelRunner::divide`]), and past those the segments left given back, which a grow creates
/// once the quota holds them; and the created segments alone, as a segment table states them
/// (the segment, its first unit and its units), which the session's head grants.
///
/// The segments past the table are the ones `segments::own` splits the whole cache into, from
/// the end of the table on: a region of one layout is split alike by every epoch that names it,
/// so a table that ends inside one of them is not this layout's.
///
/// # Errors
///
/// A table whose segments do not lie back to back from unit 0, that covers more than `units` or
/// ends inside a segment of the split; a cache that does not split; or a region that would hold
/// no segment, neither in its table nor one the quota holds now.
pub(crate) fn plane_segments(
    epoch: u64,
    (units, held): (u32, u32),
    part_bytes: u64,
    granularity: u64,
    table: &[(storage::Segment, u32, u32)],
) -> Result<(Vec<PlaneSegment>, Segments)> {
    let mapped = segments::of_table(table)?;
    let covered = segments::units(&mapped);
    if covered > units {
        bail!("the region's segment table covers {covered} units of a cache of {units}");
    }
    let split = segments::own(epoch, units, part_bytes, granularity).with_context(|| {
        format!(
            "a region of {units} units of {part_bytes} bytes a column holds no segment of whole \
             {granularity}-byte granules"
        )
    })?;

    let mut first = 0;
    let mut past = Vec::new();
    let mut stated = Vec::new();
    for mut part in split {
        if first >= covered {
            // A segment the quota does not hold yet is left given back: unmapped, and created
            // by a grow.
            part.create = first + part.segment.units <= held;
            if part.create {
                let segment = storage::Segment {
                    epoch: part.segment.epoch,
                    ordinal: part.segment.ordinal,
                };
                stated.push((segment, first, part.segment.units));
            }
            past.push(part);
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

    // The padding unit lies in the first segment, which every table holds.
    if covered == 0 && stated.is_empty() {
        bail!(
            "the GPU's quota holds none of the {units} units of the region's cache now, not one \
             segment of {} units: the device has too little free beside the epoch that serves",
            past.first().map_or(0, |part| part.segment.units)
        );
    }

    let segments = mapped.into_iter().chain(past).collect();
    Ok((segments, stated))
}

/// The bytes of the scratch page set a runner of `config` writes its paired forwards' keys and
/// values into, over a cache of `cache` (step 13), on a device whose pools map in extents of
/// `granularity` bytes: its `scratch_pages` pages, which the rows of a batch share
/// (`CausalLmRunner::scratch_pages`), and their page table, which the scratch pool holds beside
/// the runner's shadow region. Zero for a runner that computes no paired forward.
pub(crate) fn scratch_bytes(
    config: &CausalLmConfig,
    cache: &KvCacheConfig,
    granularity: u64,
) -> u64 {
    crate::memory::cache::scratch::bytes(
        &cache.layout,
        config.scratch_pages,
        cache.width,
        granularity,
    )
}

/// The extent in which the pools of `device`'s allocator map their storage, which every
/// allocation of the graph and scratch pools takes whole (`platform::Allocator::granularity`).
pub(crate) fn pool_granularity(device: DeviceId) -> Result<u64> {
    let granularity = platform::Allocator::granularity(device)
        .with_context(|| format!("reading device {}'s pool granularity", device.0))?;
    Ok(granularity as u64)
}

/// The size and alignment unit of the exportable allocations of `device`, which a pool's mapping
/// and a segment's every part take whole.
pub(crate) fn granularity(device: DeviceId) -> Result<u64> {
    let granularity = platform::DeviceAllocation::granularity(device)
        .with_context(|| format!("reading device {}'s allocation granularity", device.0))?;
    Ok(granularity as u64)
}

/// The fewest units a cache of `layout` serves with: the padding unit and one page of every cache
/// group but the state group, whose pages lie in the state region.
pub(crate) fn fewest_units(layout: &CacheLayout) -> u32 {
    1 + layout
        .groups
        .iter()
        .filter(|group| !group.state())
        .map(|group| group.units_per_page)
        .sum::<u32>()
}

/// The fewest units of the state region a cache of `layout` is built over before its division (a
/// profile run, a rank of no region): one checkpoint for a layout with a state group, none for
/// one without. A divided rank's state region is whole segments of its own.
pub(crate) fn fewest_states(layout: &CacheLayout) -> u32 {
    u32::from(layout.state_group().is_some())
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

#[cfg(test)]
mod tests {
    use nn::family::StateShape;
    use proto::request::PrefixCacheDtype;
    use storage::Segment;

    use super::*;

    /// The allocation granularity of the devices the tests split caches for.
    const GRANULE: u64 = 2 << 20;

    /// A dense cache of four layers whose planes take 32 KiB a unit, so 64 units fill a granule:
    /// 1024 units split into sixteen segments of 64.
    fn layout() -> CacheLayout {
        CacheLayout::dense(64, 4, 2, 128, PrefixCacheDtype::BFloat16)
    }

    /// The bytes one unit of the value planes takes, which its segment holds.
    fn values(layout: &CacheLayout) -> u64 {
        2 * u64::from(layout.columns) * layout.plane_bytes
    }

    #[test]
    fn the_start_creates_the_whole_segments_the_quota_holds_beside_the_unsplit_planes() {
        let layout = layout();
        let unsplit = 1024 * (layout.unit_bytes - values(&layout));
        let segment = 64 * values(&layout);

        // Three segments and most of a fourth fit beside the planes that are made whole: what is
        // left is less than one segment, one granule of each column's allocation.
        assert_eq!(
            first_units(1024, &layout, GRANULE, unsplit + 4 * segment - 1),
            192
        );
        // A quota that holds every segment holds the whole cache, and never more.
        assert_eq!(
            first_units(1024, &layout, GRANULE, unsplit + 20 * segment),
            1024
        );
        // The unsplit planes are made for every unit, or the start holds no segment at all.
        assert_eq!(first_units(1024, &layout, GRANULE, unsplit - 1), 0);
    }

    /// A hybrid cache keeps its checkpoints in the state region: a unit of the pool holds its
    /// keys and values alone, and the state region's segments are whole checkpoints, created
    /// first, as far as the quota holds them now.
    #[test]
    fn a_hybrid_cache_keeps_its_checkpoints_in_segments_of_their_own() {
        // Four attention layers and four recurrent ones, each of whose checkpoints fills one
        // 32 KiB plane: a checkpoint of four planes, 128 KiB, sixteen to a granule.
        let state = StateShape {
            layers: 4,
            fields: vec![StateField {
                name: "state",
                dtype: DType::BF16,
                dims: vec![16 << 10],
                stepwise: false,
            }],
        };
        let layout =
            CacheLayout::hybrid(64, 4, 2, 128, PrefixCacheDtype::BFloat16, &[state]).unwrap();
        let state_bytes = layout.state_bytes();
        assert_eq!(state_bytes, 4 * layout.plane_bytes);
        assert_eq!(layout.split_bytes(), values(&layout));
        assert_eq!(fewest_units(&layout), 2);
        assert_eq!(fewest_states(&layout), 1);

        // Sixteen checkpoints fill one granule: a segment of the state region.
        assert_eq!(segments::segment_units(state_bytes, GRANULE), Some(16));
        assert_eq!(first_states(64, state_bytes, GRANULE, 2 * GRANULE), 32);
        assert_eq!(first_states(64, state_bytes, GRANULE, 2 * GRANULE - 1), 16);
        assert_eq!(first_states(64, state_bytes, GRANULE, 10 * GRANULE), 64);
        assert_eq!(first_states(64, 0, GRANULE, 10 * GRANULE), 0);

        // The state region's segments are its own: a table of checkpoints is split by them.
        let (segments, stated) = plane_segments(3, (64, 32), state_bytes, GRANULE, &[]).unwrap();
        assert_eq!(segments.len(), 4);
        assert_eq!(stated.len(), 2);
        assert_eq!(stated[1].1, 16);
    }

    #[test]
    fn a_region_new_to_the_quota_creates_what_it_holds_and_leaves_the_rest_given_back() {
        let layout = layout();
        let (segments, stated) =
            plane_segments(2, (1024, 192), layout.plane_bytes, GRANULE, &[]).unwrap();

        // Every segment of the split is the cache's; the first three are created and granted,
        // the other thirteen left given back for a grow to create.
        assert_eq!(segments.len(), 16);
        let created: Vec<bool> = segments.iter().map(|part| part.create).collect();
        assert_eq!(created[..4], [true, true, true, false]);
        assert!(created[3..].iter().all(|create| !create));
        let segment = |ordinal| Segment { epoch: 2, ordinal };
        assert_eq!(
            stated,
            [
                (segment(0), 0, 64),
                (segment(1), 64, 64),
                (segment(2), 128, 64)
            ]
        );

        // The quota's figure is rounded down to whole segments, never up: 191 units hold two.
        let (_, stated) = plane_segments(2, (1024, 191), layout.plane_bytes, GRANULE, &[]).unwrap();
        assert_eq!(stated.len(), 2);

        // A region with no segment, which the quota holds none of, cannot open.
        let error = plane_segments(2, (1024, 0), layout.plane_bytes, GRANULE, &[]).unwrap_err();
        assert!(format!("{error:#}").contains("holds none"), "{error:#}");
    }

    #[test]
    fn a_region_another_epoch_named_maps_its_table_and_creates_what_the_quota_holds_past_it() {
        let layout = layout();
        let first = |ordinal| Segment { epoch: 1, ordinal };
        let table = [(first(0), 0, 64), (first(1), 64, 64)];

        // The two segments of the table are mapped; the third, which the quota holds, is created.
        let (segments, stated) =
            plane_segments(2, (1024, 192), layout.plane_bytes, GRANULE, &table).unwrap();
        assert_eq!(segments.len(), 16);
        assert_eq!(segments[0].segment.epoch, 1);
        assert_eq!(
            stated,
            [(
                Segment {
                    epoch: 2,
                    ordinal: 2
                },
                128,
                64
            )]
        );
        assert!(segments[2].create && !segments[3].create);

        // A table that covers the whole cache needs nothing of the quota.
        let whole: Vec<_> = (0..16)
            .map(|ordinal| (first(ordinal), ordinal * 64, 64))
            .collect();
        let (segments, stated) =
            plane_segments(2, (1024, 0), layout.plane_bytes, GRANULE, &whole).unwrap();
        assert_eq!(segments.len(), 16);
        assert!(stated.is_empty());
    }
}

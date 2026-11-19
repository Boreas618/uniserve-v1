//! The scheduler's KV decisions (`specs/kv-cache.md` §1.2 layer 4, §1.3): what one scheduler
//! knows of the buffer pool's cache and of its regions, and what each of its requests asks of
//! them. [`Kv`] owns that state and those rules; the scheduler owns the requests' lives, the
//! queue pair, the executor and the policy, and states to [`Kv`] only what a decision reads:
//!
//! - **In.** The request a decision is for (its tokens, progress, placement and tables,
//!   `Request`), the worker groups it may run on (`WorkerGroup`: the cache groups their workers
//!   state and the regions the session gives each class there), the scheduler's resolved
//!   configuration and the step's [`Budget`], every completion the buffer pool wrote, with the
//!   request of the owner it acts for ([`Kv::learn`]), every request the scheduler submits
//!   ([`Kv::submitted`]), and the capacity words of the regions ([`Kv::set_words`]).
//! - **Out.** The hit a request's prefix finds and the requests to the buffer pool that hold it
//!   (`Pin`, `Load`, `Clone`, `Prefetch`, `Allocate`, `Release`, `Commit`, `Store`, `Victims`,
//!   `List`), which the scheduler submits; the units each region has available to allocations;
//!   the pages a handoff's KV program decides over ([`scheduling::kv::Handoff`]); and what the
//!   policy is told of the cache (`scheduling::Event`), appended to the sink it is handed.
//!
//! Nothing here touches the queue pair, waits, or reads another scheduler's state: every
//! decision is a function of the state below and of what the call states, so a [`Kv`] is built
//! and tested without a scheduler ([`Kv::new`]).
//!
//! - `cache.rs`: the [`Cached`] row of every cached block the scheduler knows of, the eviction
//!   order of each region, the write policy's refusals, and what each completion states of them.
//! - `tables.rs`: a request's tables, from its keys and the hit its admission rounds find to
//!   the blocks its calls need, commit and give up.
//! - `handoff.rs`: the pages of a handoff's import as the decode side's tiers hold them.
//! - this file: the session's classes and regions, and each region's units.

mod cache;
mod handoff;
mod tables;

use std::collections::{BTreeSet, HashMap, HashSet};

use proto::config::SchedulerConfig;
use storage::Key;
use storage::runtime::{self, Tier};

use super::events::Events;
use super::{Budget, ENCODER_IDENTITIES, Error, WorkerGroup, agreed, differ};

pub(crate) use cache::{Cached, LOW_WATERMARK, covered, stamped_groups};
pub(super) use cache::{DEVICE, HOST, REMOTE};
pub(super) use tables::{
    ahead, count_asked, draft_class, draft_order, draft_tokens, fit_chunk, holds_blocks_for, kept,
    kv_owner, next_call, restored, resume, settled, shapes, state_shape,
};

/// What the scheduler states of itself to a KV decision that plans a request's calls, borrowed
/// for the call: the worker groups the request may run on, the scheduler's resolved
/// configuration (its drafts per verification and its copy timeouts), the step's budget (its
/// chunk and tokens, and whether it caches prefixes and offloads to the host), and the policy's
/// `prefetch_min_tokens`, the least deepening a prefetch the policy did not name pays for.
#[derive(Clone, Copy)]
pub(crate) struct Step<'a> {
    pub groups: &'a [WorkerGroup],
    pub config: &'a SchedulerConfig,
    pub budget: &'a Budget,
    pub prefetch_min_tokens: u32,
}

/// The scheduler's KV decisions and their state (`specs/kv-cache.md` §1.3).
///
/// A region is named in completions and requests by its number at the buffer pool, and indexed
/// here, in every per-region field, by its place in the session's regions (`Open::regions`),
/// which is how the layout of a group and a class names it too.
///
/// Invariants: a block is in its region's `order` exactly when it is in `cache`, on that region's
/// tier, with no pin of the scheduler's requests; `ordered` is the units of each `order`;
/// `installed` names a head of each `order` as the scheduler last installed it.
pub(crate) struct Kv {
    /// The classes of the session; a request names a class by its index here.
    classes: Vec<runtime::Class>,
    /// The class of stored trajectories, as the session states it (`Open::trajectory`).
    trajectory: u8,
    /// Per region of the session: the number completions and requests name it by, and its tier
    /// and unit size as the layout states them.
    regions: Vec<(u8, runtime::Region)>,
    /// Per region of the session: the units its numbering holds past its reserved ones, as the
    /// buffer pool numbers it (`Opened::limits`), the most a region resized to none holds.
    numbered: Vec<u32>,
    /// The regions the scheduler's queue pair has capacity words of: each one's number and its
    /// index in `regions`, in the order the session states them (`Registers::capacity`).
    used: Vec<(u8, usize)>,
    /// The bytes of the largest record of the queue pair's submission queue, which bound the
    /// keys of one `Victims` or `Evict`.
    record: usize,

    /// Per region, its units, from the answer to `Open` and from `Resized`.
    capacity: Vec<u32>,
    /// Per region, the units the engine last resized it to, which bound what it holds from then
    /// on (`resize.rs`): while a shrink has stopped short of them, the scheduler installs the
    /// whole of the region's eviction order, takes none of the units past them for its own
    /// allocations, and shrinks the region again as units come free; they are the units a
    /// request's admission is judged against ([`Kv::declared`]). `None` before the engine
    /// resized the region and from a drain on.
    bounds: Vec<Option<u32>>,
    /// Per region, the units an allocation of the scheduler can take: the buffer pool's word,
    /// less the units of the requests the scheduler has not seen answered, and with the units of
    /// `order` beyond the head it installed ([`Kv::read_capacity`]).
    available: Vec<u32>,
    /// Per region, the units of retired blocks the buffer pool reclaims once the batches that may
    /// read them have ended.
    retiring: Vec<u32>,
    /// Per region, its three capacity words as the scheduler last read them: `available`,
    /// `retiring` and `evictable` (every shared block without a pin there, in any eviction order
    /// or none). Zero for a region the queue pair has no words of.
    words: Vec<[u32; 3]>,

    /// Every cached block the scheduler knows of, by [`Cached::id`].
    cache: HashMap<u64, Cached>,
    /// Per region, the cached blocks no request of the scheduler holds, by last use and id: the
    /// eviction order it installs.
    order: Vec<BTreeSet<(u64, u64)>>,
    /// Per region, the units of the blocks in `order`.
    ordered: Vec<u32>,
    /// Per region, the numbers of the keys of the last `Victims` the scheduler submitted there:
    /// the head of `order` it installed.
    installed: Vec<Vec<u64>>,
    /// The blocks a request gave up before the buffer pool answered their `Commit`, by
    /// [`Cached::id`] of the key: the last use each enters the eviction order with once its
    /// `Committed` is read, as `unpin` ranked it among its request's blocks then. An entry leaves
    /// when its `Commit` is answered.
    given_up: HashMap<u64, u64>,
    /// The `Commit`s the scheduler submitted whose answers it has not read, by [`Cached::id`] of
    /// their keys, with how many of them name each.
    committing: HashMap<u64, u32>,
    /// Per region, the `Store`s to the host into it that stored a block (answered `Stored` after
    /// their copy ran): a block whose own `Store` stored nothing waits for this count, among
    /// others, to move before it is offered to the host again (`Cached::refused`).
    stores: Vec<u64>,
    /// The blocks, by [`Cached::id`], whose last `Store` to the host stored nothing: each step's
    /// `offload` offers each again once something that could change that answer has happened. A
    /// block leaves when it is offered again, has a host copy, or has no device copy.
    unstored: HashSet<u64>,
    /// The `List`s submitted, and those answered, which the buffer pool answers in the order it
    /// takes them: a resumed request waits for one asked after it arrived.
    lists_asked: u64,
    lists_answered: u64,

    /// Per cache group, in class order, the seed the key chains of its blocks are rooted at; a
    /// request's `CachePolicy::state_seeds` override it. Empty until stated, which roots every
    /// chain at the zero key, the seed of an empty identity.
    seeds: Vec<Key>,
    /// The identity of the weights that compute the scheduler's encoder outputs, which keys them
    /// (`encoder_output`): what its encoding groups state (`Capacity::encoder_identity`), empty
    /// when none does.
    encoder_identity: String,
    /// The seed of the keys of stored trajectories (`latent_content`): what its denoising groups
    /// state (`Capacity::latent_seed`), 0 when none does.
    latent_seed: u64,

    /// The KV events published of the cached blocks the scheduler commits (`kv_events_config`).
    events: Events,
}

impl Kv {
    /// The KV decisions of the scheduler whose session the engine opened with `layout`, which
    /// `opened` answered, over its `groups` and its queue pair's capacity words at `registers`,
    /// whose records hold at most `record` bytes; `rank` is the scheduler's data-parallel rank,
    /// which its KV events state until numbered. It knows no cached block yet.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidRequest`] when two groups state different page seeds, encoder identities
    /// or latent seeds: one key chain serves every group, so a request's blocks, encoder outputs
    /// and trajectory are found under one key whichever of its groups computed them. A group that
    /// states none (no KV cache, no encoder, no denoiser) agrees with every other.
    pub(crate) fn new(
        layout: &runtime::Open,
        opened: &runtime::Opened,
        registers: &runtime::Registers,
        record: usize,
        groups: &[WorkerGroup],
        rank: u32,
    ) -> Result<Self, Error> {
        let stated = || groups.iter().map(|group| &group.capacity);
        let seeds = agreed(stated().map(|capacity| &capacity.seeds))
            .ok_or_else(|| differ("page seeds for their cache groups"))?;
        let encoder_identity = agreed(stated().map(|capacity| &capacity.encoder_identity))
            .ok_or_else(|| differ(ENCODER_IDENTITIES))?;
        let latent_seed = agreed(stated().map(|capacity| &capacity.latent_seed))
            .ok_or_else(|| differ("latent seeds for their stored trajectories"))?;

        let regions: Vec<(u8, runtime::Region)> = opened
            .regions
            .iter()
            .zip(&layout.regions)
            .map(|(&(number, _), region)| (number, region.clone()))
            .collect();
        let capacity: Vec<u32> = opened.regions.iter().map(|&(_, units)| units).collect();
        // A pool that states no limit numbers the region as the layout does.
        let numbered = regions
            .iter()
            .enumerate()
            .map(|(at, (_, region))| {
                let limit = opened.limits.get(at).copied().unwrap_or(region.limit);
                limit.saturating_sub(region.reserved)
            })
            .collect();
        let count = regions.len();
        let used = registers
            .capacity
            .iter()
            .filter_map(|&(number, _)| {
                let index = regions.iter().position(|(stated, _)| *stated == number)?;
                Some((number, index))
            })
            .collect();

        Ok(Self {
            classes: layout.classes.clone(),
            trajectory: layout.trajectory,
            regions,
            numbered,
            used,
            record,
            available: capacity.clone(),
            capacity,
            bounds: vec![None; count],
            retiring: vec![0; count],
            words: vec![[0; 3]; count],
            cache: HashMap::new(),
            order: vec![BTreeSet::new(); count],
            ordered: vec![0; count],
            installed: vec![Vec::new(); count],
            given_up: HashMap::new(),
            committing: HashMap::new(),
            stores: vec![0; count],
            unstored: HashSet::new(),
            lists_asked: 0,
            lists_answered: 0,
            seeds,
            encoder_identity,
            latent_seed,
            events: Events::new(rank),
        })
    }

    /// Records a request the scheduler submits to the buffer pool, before it is written: a
    /// `Commit` is unanswered until its answer is learned, and a `List` is asked.
    pub(crate) fn submitted(&mut self, request: &runtime::Request) {
        match request {
            runtime::Request::List => self.lists_asked += 1,
            runtime::Request::Commit { key, .. } => {
                *self.committing.entry(Cached::id(key)).or_default() += 1;
            }
            _ => {}
        }
    }

    /// The `List`s asked so far: a request that arrives now waits, where it waits for one, for
    /// a `List` asked after this many ([`Kv::awaits_listing`]).
    pub(crate) fn lists_asked(&self) -> u64 {
        self.lists_asked
    }

    /// The KV events the scheduler publishes.
    pub(crate) fn events(&mut self) -> &mut Events {
        &mut self.events
    }
}

/// The session's classes and regions (`Open::classes`, `Open::regions`).
impl Kv {
    /// The classes of the session, by index.
    pub(crate) fn classes(&self) -> &[runtime::Class] {
        &self.classes
    }

    /// The class of stored trajectories.
    pub(crate) fn trajectory(&self) -> u8 {
        self.trajectory
    }

    /// The seed of the keys of stored trajectories.
    pub(crate) fn latent_seed(&self) -> u64 {
        self.latent_seed
    }

    /// The regions of the session, by index, each with the number the buffer pool names it by.
    pub(crate) fn regions(&self) -> &[(u8, runtime::Region)] {
        &self.regions
    }

    /// The regions the queue pair has capacity words of: each one's number and index.
    pub(crate) fn used(&self) -> &[(u8, usize)] {
        &self.used
    }

    /// The index of the region the buffer pool numbers `number`.
    pub(crate) fn region_index(&self, number: u8) -> Option<usize> {
        self.regions
            .iter()
            .position(|(stated, _)| *stated == number)
    }

    /// Whether region `region` (an index) lies on `tier`.
    pub(crate) fn on_tier(&self, region: usize, tier: Tier) -> bool {
        self.regions
            .get(region)
            .is_some_and(|(_, stated)| stated.tier == tier)
    }

    /// The units of one block of `class`: the class's block size, or `units` for a sized class.
    pub(crate) fn block_units(&self, class: u8, units: u32) -> Option<u32> {
        let stated = self.classes.get(usize::from(class))?.units;
        Some(if stated > 0 { stated } else { units })
    }

    /// The units a block of `class` takes in `region` (by index): a host copy's configured
    /// size, else the class's block size; 0 for a sized class, whose blocks state their own.
    pub(crate) fn class_units(&self, class: u8, region: usize) -> u32 {
        let Some(stated) = self.classes.get(usize::from(class)) else {
            return 0;
        };
        match stated.host {
            Some((host, units)) if usize::from(host) == region && units > 0 => units,
            _ => stated.units,
        }
    }

    /// The host region (by index) of `class`, and the units one host copy of its blocks takes
    /// there: the size the class states for its host copies, else its block size, or `units`
    /// for a sized class. `None` for a class with no host region.
    pub(crate) fn host_block(&self, class: u8, units: u32) -> Option<(usize, u32)> {
        let (region, _) = self.classes.get(usize::from(class))?.host?;
        let region = usize::from(region);
        let each = match self.class_units(class, region) {
            0 => units,
            each => each,
        };
        Some((region, each))
    }

    /// The regions, as indices, the blocks of `class` of an owner on `group` are taken from.
    pub(crate) fn class_regions(group: &WorkerGroup, class: u8) -> Vec<usize> {
        let regions = group.regions(class);
        regions.iter().map(|&region| usize::from(region)).collect()
    }

    /// The runs one block of `class` has for an owner on `group`, and its units.
    pub(crate) fn block_shape(
        &self,
        group: &WorkerGroup,
        class: u8,
        units: u32,
    ) -> Option<(usize, usize)> {
        let runs = Self::class_regions(group, class).len().max(1);
        let units = usize::try_from(self.block_units(class, units)?).ok()?;
        Some((runs, units.max(1)))
    }

    /// The units of the session's region `region` (an index into its regions) that hold `bytes`
    /// on one rank, a part of each unit (`Region::part_bytes`): at least one, since a block has a
    /// size.
    pub(crate) fn units_in(&self, region: u8, bytes: u64) -> Option<u32> {
        let (_, stated) = self.regions.get(usize::from(region))?;
        let units = bytes.div_ceil(stated.part_bytes.max(1)).max(1);
        u32::try_from(units).ok()
    }

    /// The fields of the largest `Copy` answering a `Store` to the host of `keys` from `group`
    /// (BP §11): per key, the runs of its device block in the group, one host run, and an
    /// evicted key per unit of its host copy, through `copy` (the record's fields of a copy of
    /// that many keys, source runs, destination runs and evicted keys). `None` when a key's
    /// class or host copy size is not known.
    pub(crate) fn store_bytes(
        &self,
        group: &WorkerGroup,
        keys: &[Key],
        copy: &impl Fn(usize, usize, usize, usize) -> usize,
    ) -> Option<usize> {
        let (mut from, mut evicted) = (0usize, 0usize);
        for key in keys {
            let class = self.cache.get(&Cached::id(key))?.class;
            let (_, host) = self.classes.get(usize::from(class))?.host?;
            let units = if host > 0 {
                host
            } else {
                self.block_units(class, 0).filter(|units| *units > 0)?
            };
            from = from.saturating_add(group.regions(class).len().max(1));
            evicted = evicted.saturating_add(units as usize);
        }
        Some(copy(keys.len(), from, keys.len(), evicted))
    }
}

/// Each region's units (SD §4.1, §4.5; `resize.rs`).
impl Kv {
    /// The units region `region` (an index) holds, as the session's `Open` and its last
    /// `Resized` stated them.
    pub(crate) fn capacity(&self, region: usize) -> u32 {
        self.capacity.get(region).copied().unwrap_or(0)
    }

    /// The most units of region `region` (an index) a shrink to `units` finds in its tail, the
    /// units past its first `units`: those its numbering holds there, and no more than the region
    /// holds.
    pub(crate) fn tail_units(&self, region: usize, units: u32) -> u32 {
        let numbered = self.numbered.get(region).copied().unwrap_or(u32::MAX);
        numbered.saturating_sub(units).min(self.capacity(region))
    }

    /// A `Resized` stated that region `region` (an index) holds `units`.
    pub(crate) fn resized(&mut self, region: usize, units: u32) {
        if let Some(capacity) = self.capacity.get_mut(region) {
            *capacity = units;
        }
    }

    /// The units an allocation of the scheduler can take from region `region` (an index) now.
    pub(crate) fn available(&self, region: usize) -> u32 {
        self.available.get(region).copied().unwrap_or(0)
    }

    /// `units` of region `region` (an index) are given to a request the step asks for: they are
    /// not available to the step's later requests.
    pub(crate) fn take(&mut self, region: usize, units: u32) {
        if let Some(available) = self.available.get_mut(region) {
            *available = available.saturating_sub(units);
        }
    }

    /// The units of region `region` (an index) that retire.
    pub(crate) fn retiring(&self, region: usize) -> u32 {
        self.retiring.get(region).copied().unwrap_or(0)
    }

    /// Word `at` of the capacity words of region `region` (an index) as last read: 0 its
    /// `available` units, 1 its `retiring` units, 2 its `evictable` units.
    pub(crate) fn word(&self, region: usize, at: usize) -> u32 {
        self.words
            .get(region)
            .and_then(|words| words.get(at))
            .copied()
            .unwrap_or(0)
    }

    /// The capacity words of region `region` (an index) read now: `available`, `retiring` and
    /// `evictable`.
    pub(crate) fn set_words(&mut self, region: usize, words: [u32; 3]) {
        if let Some(read) = self.words.get_mut(region) {
            *read = words;
        }
    }

    /// Sets each used region's `available` and `retiring` from the words last read
    /// ([`Kv::set_words`]): `available` is the buffer pool's word raised by the units of `order`
    /// beyond the installed head, which the buffer pool may evict once a later `Victims` names
    /// them, and lowered by the units past the region's bound (its shrink's to take, so no
    /// allocation holds units past the bound) and by `asked`, per region the units the
    /// scheduler's requests not yet answered take, which the buffer pool's word may not show yet.
    pub(crate) fn read_capacity(&mut self, asked: &[(usize, u32)]) {
        for at in 0..self.used.len() {
            let (_, region) = self.used[at];
            let [available, retiring, _] = self.words[region];
            self.available[region] = available
                .saturating_add(self.beyond_head(region))
                .saturating_sub(self.excess(region));
            self.retiring[region] = retiring;
        }
        for &(region, units) in asked {
            self.take(region, units);
        }
    }

    /// The units of `order[region]` beyond the head installed there, which the buffer pool's
    /// `available` word does not count: it evicts them once a later `Victims` names them.
    pub(crate) fn beyond_head(&self, region: usize) -> u32 {
        let (Some(order), Some(installed)) = (self.order.get(region), self.installed.get(region))
        else {
            return 0;
        };
        let kept = |id: &u64| {
            self.cache
                .get(id)
                .filter(|row| order.contains(&(row.last_used, *id)))
        };
        let head = installed
            .iter()
            .filter_map(kept)
            .map(|row| self.class_units(row.class, region));
        self.ordered.get(region).map_or(0, |units| {
            units.saturating_sub(head.fold(0, u32::saturating_add))
        })
    }

    /// Bounds region `region` (an index) to `units` from here on, as the engine resized it.
    pub(crate) fn bound(&mut self, region: usize, units: u32) {
        if let Some(bound) = self.bounds.get_mut(region) {
            *bound = Some(units);
        }
    }

    /// The bound of region `region` (an index), `None` while it has none.
    pub(crate) fn bound_of(&self, region: usize) -> Option<u32> {
        self.bounds.get(region).copied().flatten()
    }

    /// Ends every bound: a drain sizes the regions from here on.
    pub(crate) fn end_bounds(&mut self) {
        self.bounds.fill(None);
    }

    /// The units region `region` (an index) is declared to hold: the units the engine last
    /// resized it to, or, before any resize and from a drain on, the units its last `Resized`
    /// (or the session's `Open`) stated. A request is too large for the region only beyond these.
    ///
    /// The region's current units may lie on either side of them for a while: a shrink that
    /// stopped short holds more until the units held past the bound come free, and a grow that
    /// stopped short holds fewer until the memory it waits for is granted to the region's space,
    /// after which the engine resizes the region again. A request that fits the declared units
    /// waits through either; refusing it against the current units would refuse a request the
    /// region holds once its resize completes.
    pub(crate) fn declared(&self, region: usize) -> u32 {
        self.bound_of(region)
            .unwrap_or_else(|| self.capacity(region))
    }

    /// The units region `region` (an index) holds past its bound, as its last `Resized` stated
    /// them: what its shrink is still to take. 0 for a region with no bound.
    pub(crate) fn excess(&self, region: usize) -> u32 {
        match (self.bound_of(region), self.capacity.get(region)) {
            (Some(bound), Some(&units)) => units.saturating_sub(bound),
            _ => 0,
        }
    }

    /// The classes whose blocks `region` (by index) holds on the scheduler's side: as a device
    /// region of one of `groups`, or as the class's host region.
    pub(crate) fn classes_in<'a>(
        &'a self,
        region: usize,
        groups: &'a [WorkerGroup],
    ) -> impl Iterator<Item = u8> + 'a {
        let host = move |class: &runtime::Class| {
            class
                .host
                .is_some_and(|(host, _)| usize::from(host) == region)
        };
        let device = move |class: u8| {
            groups.iter().any(|group| {
                group
                    .regions(class)
                    .iter()
                    .any(|&r| usize::from(r) == region)
            })
        };
        (0..=u8::MAX)
            .zip(&self.classes)
            .filter(move |&(class, stated)| host(stated) || device(class))
            .map(|(class, _)| class)
    }
}

/// The state a scheduler's test sets up directly, where its case starts from a cache or regions
/// no sequence of completions on its bench reaches as briefly.
#[cfg(test)]
impl Kv {
    /// Every region's units available to allocations.
    pub(crate) fn available_mut(&mut self) -> &mut Vec<u32> {
        &mut self.available
    }

    /// Every region's retiring units.
    pub(crate) fn retiring_mut(&mut self) -> &mut Vec<u32> {
        &mut self.retiring
    }

    /// The session's classes.
    pub(crate) fn classes_mut(&mut self) -> &mut Vec<runtime::Class> {
        &mut self.classes
    }

    /// The class of stored trajectories.
    pub(crate) fn trajectory_mut(&mut self) -> &mut u8 {
        &mut self.trajectory
    }

    /// Every region's count of `Store`s to the host that stored a block.
    pub(crate) fn stores_mut(&mut self) -> &mut Vec<u64> {
        &mut self.stores
    }

    /// `row` as a row of `cache`, in no eviction order.
    pub(crate) fn insert(&mut self, row: Cached) {
        self.cache.insert(Cached::id(&row.key), row);
    }
}

#[cfg(test)]
pub(super) mod tests {
    //! A [`Kv`] built alone, over the session of `scheduler::tests::bench`: group A's blocks of
    //! class `KV` lie in the region numbered 4, group B's in region 5, the product class in
    //! region 6 and the slot rows in region 7.

    use super::Kv;
    use crate::scheduler::WorkerGroup;
    use crate::scheduler::tests::session;

    /// The KV decisions of the bench's session and its groups, with records of `record` bytes.
    pub(crate) fn kv(record: usize) -> (Kv, Vec<WorkerGroup>) {
        let (layout, opened, groups) = session();
        let registers = &opened.registers[0];
        let kv = Kv::new(&layout, &opened, registers, record, &groups, 0).unwrap();
        (kv, groups)
    }
}

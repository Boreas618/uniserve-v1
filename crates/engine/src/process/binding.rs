//! The weight binding of a serving epoch: the operations thread's changes to what its ranks hold
//! on their devices inside the epoch, weights above all. Another checkpoint's weights are staged
//! beside the serving batches and bound at a batch boundary in place of the ones they replace; the
//! KV planes' segments, the graph pools, LoRA adapters, profiling sessions and the process's
//! reservation in its GPUs' quotas change through the same ranks.
//!
//! Every change reaches a device as a control operation of the rank's worker
//! ([`Worker::control`]), which runs between the jobs submitted before and after it, once the
//! device has passed everything the runners issued: nothing but the worker's thread issues on
//! the device, and no job in flight shares a mapping change. An operation is handed to every
//! rank before any is waited for, so the devices perform it at once, and it returns once every
//! rank's has ended.
//!
//! The work comes in three kinds, which differ in who may run them when:
//!
//! - **At a batch boundary**: binding staged weights in place ([`WeightBinding::update_weights`],
//!   [`WeightBinding::commit_experts`]), suspending and restoring the graph pools, releasing and
//!   creating the KV planes' segments, and the kernels the cache and the state launch, each
//!   rank's from the generation its graphs serve ([`WeightBinding::set_kernels`]). A control
//!   operation is ordered against its own worker's jobs only, so the engine first pauses the
//!   schedulers whose batches read what changes: every rank of a group then changes between the
//!   same two batches.
//! - **Beside the serving batches**: staging a load of another checkpoint of the same model, or
//!   of another arrangement of its experts ([`WeightBinding::pair`], [`WeightBinding::load`]),
//!   published at the Storage Server while requests go on; LoRA adapters; profiling sessions,
//!   which each rank's worker starts and ends between two of its batches.
//! - **The quota**: what the process holds on each device beside its pools and the Storage
//!   Server's storages, reserved in the GPU's quota ([`WeightBinding::reserve_serving`]) with
//!   what a profiling session's collectors hold.
//!
//! A weight is never written after it is published, by anyone. An update changes which published
//! allocation an address maps: the graphs captured over that address stay valid, the mapping
//! stays read-only, and the allocation is the one a relaunch on the new target finds under its
//! own key.
//!
//! ## The KV planes' memory
//!
//! The buffer pool's tables say which units of a region's memory space a request holds; the
//! memory behind the units is the ranks' segments of the KV planes, which each rank creates in
//! its own context (`specs/buffer-pool.md` §14.1). Memory changes by whole segments, around the
//! tables' changes, and the head of the epoch's session states each change to the Storage Server,
//! since only the head knows when every rank that maps a region has made it. A region's ranks are
//! those of the worker groups that state it, which the segment operations name. The weight
//! binding is the ranks' side of each sequence:
//!
//! 1. **Release from segment `k` on**, for a region of another layout to grow into the memory:
//!    the schedulers `Resize` their regions below the first unit of segment `k`; the head sends
//!    `Withdraw` for each segment from `k` on, which takes its units out of the space's free
//!    ranges; then [`WeightBinding::release_segments`]: every rank of the region unmaps them and
//!    ends its retention, and the GPU's quota usage drops with the last rank. A start whose ranks
//!    are warm releases so the tail segments its quotas no longer hold beside the warm processes,
//!    as many as [`WeightBinding::warm_segments`] leaves, before its regions hold any unit.
//! 2. **Create given-back segments**, for a region to grow past the space's free ranges, as many
//!    as [`WeightBinding::fitting_segments`] finds room for: [`WeightBinding::create_segments`]:
//!    every rank of the region creates them again at their addresses and resets their units; the
//!    head sends `Grant` for each segment the ranks hold now beyond those the space held; then
//!    the schedulers `Resize` their regions into them.
//! 3. **Sleep**: the schedulers detach every owner and `Resize` every device region to nothing,
//!    having stored what the write policy keeps; the head withdraws every segment;
//!    [`WeightBinding::sleep`] releases them all and suspends the graph pools.
//! 4. **Wake**: [`WeightBinding::restore_graphs`] restores the graph pools; then, as a grow, every
//!    rank of a region creates the segments it gave back that its quota holds beside its warm
//!    process, which its start reserved, and the head grants them; the schedulers grow their
//!    regions.
//!
//! Each returns its ranks' segment tables ([`crate::memory::cache::info::CacheTable`]), whose
//! segment `i` holds the units from the sum of the units of the segments before it. An epoch switch
//! changes no plane: two epochs of one layout name the same regions and map the same segments, and
//! nothing of the cache is handed over or adopted (`specs/updates.md` §3.4).
//!
//! ## Ranks on other hosts
//!
//! A worker group may have ranks in the epoch's other main processes (`crate::executor::peers`).
//! The scheduler node's binding holds the links to them, each with the groups it holds ranks of,
//! and every operation that acts on ranks acts on theirs too: it hands the operation to this
//! process's ranks and sends it to each link that holds a rank it names
//! ([`DataOperation`]), which the peer runs on its own binding, then awaits every part and joins
//! the values. Segment tables are listed in rank order and must agree where the ranks create
//! segments; rooms are listed, and the segments that fit and the graph bytes given back are the
//! smallest over every rank; reports and results are listed in rank order. A boundary
//! operation is ordered against the group's batches as this process's is: the schedulers are
//! paused, and no batch travels to a peer until it ended there. A weight load's replacement stays
//! in each process: a peer pairs, loads, stages and discards its own as this process does.
//! An expert redistribution's stages run on each process's own ranks as the daemon's operations
//! reach it ([`WeightBinding::hook_peers`]); its commit acts on every rank.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

use crate::executor::peers::messages::DataOperation;
use crate::executor::peers::{Answer, Remote, operated};
use crate::memory::cache::KvCacheMemory;
use crate::memory::cache::info::CacheTable;
use crate::memory::client::{StorageClient, quota_limit};
use crate::memory::division::{self, process_bytes};
use crate::memory::graph_pool::GraphPool;
use crate::memory::reload::{MetaWeights, Reload, paired};
use crate::memory::state::StateMemory;
use crate::memory::weights::WeightStore;
use crate::profiling::session::Change;
use crate::worker::completion::Outcome;
use nn::WeightKey;
use platform::{DeviceId, Nvml, Stream};
use proto::request::wire::{DeviceMemory, PoolMemory};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use service::lock;
use storage::Item;
use storage::runtime::GroupId;

use crate::runner::Runner;
use crate::worker::{Error, Result, Worker};

/// The name [`WeightBinding::memory`] reports the planes of a device's KV regions under, beside the
/// pools of its allocators (`graph`, `scratch`, `resident`): the segments the ranks map of the
/// value and state planes, with the flags, scales and index keys beside them.
pub(crate) const KV_POOL: &str = "kv";

/// What a load of the served model reads: which checkpoint, and which expert sits in which slot.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Target {
    /// The checkpoint's directory.
    pub checkpoint: PathBuf,
    /// Each routed expert layer's logical expert ids in physical slot order, by module path. A
    /// layer this map does not name keeps the checkpoint's order.
    pub experts: BTreeMap<String, Vec<u32>>,
}

/// How a rank loads its model: the model crates are the executable's, so this crate reaches them
/// through what the executable hands it.
pub(crate) trait ModelLoad: Send + Sync {
    /// Runs the model's load of `target` through `weights`, which every weight goes through: the
    /// rank's store, a store on the meta device that reads no payload, or a second load beside
    /// the weights that serve.
    ///
    /// # Errors
    ///
    /// Why the checkpoint cannot be this model's, as a sentence.
    fn load(
        &self,
        target: &Target,
        weights: &dyn nn::RetainedWeights,
    ) -> std::result::Result<(), String>;

    /// The routed expert layers this rank's model holds, by module path, each with the experts its
    /// router names. A model without routed experts states none.
    fn expert_layers(&self) -> BTreeMap<String, u32>;
}

/// The devices a release of segments freed memory on ([`WeightBinding::release_segments`]), per
/// process that holds ranks of its groups: this process's (`None`) and each peer's by its first
/// rank, as the CUDA ordinals of the devices of the ranks that gave segments back.
pub(crate) type Released = Vec<(Option<u32>, Vec<u32>)>;

/// What the weight binding holds of one rank: the owners of its device's storage, which the rank's
/// start opened, and the worker every change of them goes through.
pub(crate) struct Rank {
    /// The rank's number in its epoch, as a profiling session's `workers` name it.
    pub rank: u32,
    /// The rank's worker. Weak, because the epoch's close joins the worker's thread and owns
    /// the last handle.
    pub worker: Weak<Worker>,
    /// The rank's device.
    pub device: DeviceId,
    /// The fraction of the device its epoch's quota of the GPU is limited to, which a
    /// reservation states with it (`WorkerGroupConfig::device_fraction`).
    pub fraction: f64,
    /// The rank's Storage Server client.
    pub client: Arc<StorageClient>,
    /// The weights the rank serves from, which a second load publishes beside.
    pub weights: Arc<WeightStore>,
    /// The rank's KV cache, on a rank that keeps one.
    pub cache: Option<Arc<Mutex<KvCacheMemory>>>,
    /// The rank's recurrent state, for a model that keeps one.
    pub state: Option<Arc<Mutex<StateMemory>>>,
    /// The device's graph, scratch and resident pools.
    pub graphs: Arc<Mutex<GraphPool>>,
    /// A stream of the device that only the weight binding's control operations issue on: the
    /// resets of the units of segments created again.
    pub stream: Stream,
    /// The device bytes of the rank's communicators that NVML attributes to no process, as their
    /// owner states them: what a measurement of the process adds to NVML's figure.
    pub unattributed: u64,
    /// How the rank loads its model.
    pub model: Arc<dyn ModelLoad>,
    /// The weights the rank serves, in its load's order, each with whether the model writes it
    /// after the load.
    pub serving: Vec<(WeightKey, bool)>,
    /// Whether the rank computes encoder outputs, which the identity of its group's encoding
    /// ranks' weights keys ([`encoder_identities`]).
    pub encodes: bool,
}

impl Rank {
    /// The rank's worker, until the epoch closed it.
    fn worker(&self) -> Result<Arc<Worker>> {
        self.worker
            .upgrade()
            .ok_or(Error::State("the rank's worker has closed"))
    }

    /// Whether the rank is one of the worker groups `groups`, as its worker's block tables
    /// number its group in the session. A rank whose worker closed is of none.
    fn of(&self, groups: &[GroupId]) -> bool {
        self.worker
            .upgrade()
            .is_some_and(|worker| groups.contains(&worker.group()))
    }
}

/// What a control operation computes on one rank: called on the rank's worker thread with its
/// runners, once the device has passed everything they issued.
type Work<T> = Box<dyn FnOnce(&mut [Box<dyn Runner>]) -> Result<T> + Send>;

/// What a segment decision reads of one rank whose KV planes are split: its GPU's identity, its
/// device's room, its segment table, and the device bytes one unit of its planes takes, which a
/// unit's release gives back.
type SegmentFacts = ([u8; 16], Room, CacheTable, u64);

/// One rank's share of a replacement: which weights it moves, and what that takes.
struct Share {
    /// Each weight that serves with the weight that takes its place.
    pairs: Vec<(WeightKey, WeightKey)>,
    /// The target's load, in order: the rank's serving order once the replacement is in place.
    order: Vec<(WeightKey, bool)>,
    /// Device bytes the replacements take beside the weights that serve.
    bytes: u64,
}

/// A load of another target on every rank, from its pairing to the boundary that puts it in
/// place, or to its discard.
pub(crate) struct Replacement {
    target: Target,
    ranks: Vec<Share>,
    /// The most device bytes a rank of the peers needs beside what it serves: each peer keeps
    /// its own share of the replacement.
    peers: u64,
    /// Whether the new weights are published yet, so a discard knows what there is to release.
    loaded: bool,
}

impl Replacement {
    /// The most device bytes any rank needs beside what it serves.
    pub(crate) fn bytes(&self) -> u64 {
        self.ranks
            .iter()
            .map(|rank| rank.bytes)
            .max()
            .unwrap_or(0)
            .max(self.peers)
    }
}

/// The room one device has beside what it holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Room {
    /// Bytes the deployment's quota of the GPU leaves, the Transfer Server's context counted
    /// whether the server holds it now or not (`division::unheld_context`).
    pub quota: u64,
    /// Bytes the device has free.
    pub free: u64,
    /// Bytes of the quota that other epochs of the deployment hold and the rank's does not
    /// (`division::held_elsewhere`), which return to the quota and the device as those epochs
    /// drain: what the rank's epoch holds beside the rest as if it served the GPU alone.
    pub elsewhere: u64,
}

impl Room {
    /// The room of a device whose deployment's quota is `quota` and which has `free` bytes free,
    /// for a rank of `epoch`: what the quota leaves, what the device has free, and what other
    /// epochs of the deployment hold of the quota. A quota of no limit leaves what the device has.
    pub(crate) fn of(quota: &storage::Quota, free: u64, epoch: u64) -> Self {
        // The Transfer Server's context is counted whether it holds it now or not: a storage that
        // took its room would have the quota refuse the context, and the task that needs it.
        let reserved = quota
            .usage()
            .saturating_add(division::unheld_context(quota));

        Self {
            quota: quota
                .limit
                .map_or(free, |limit| limit.saturating_sub(reserved)),
            free,
            elsewhere: division::held_elsewhere(quota, epoch),
        }
    }

    /// Whether a load of `bytes` fits here once `given_back` bytes of the device leave. The graph
    /// pools a process gives back free the device and the quota alike, because a rank that serves
    /// reserves them in the quota and a suspension takes them out of it.
    pub(crate) fn holds(self, bytes: u64, given_back: u64) -> bool {
        bytes <= self.quota.min(self.free).saturating_add(given_back)
    }
}

/// What one rank's process holds in its GPU's quota beside its pools and the Storage Server's
/// storages, as `Item::Process`.
struct Reservation {
    /// The process as measured when the rank started to serve.
    process: u64,
    /// What the collectors of the profiling session that runs hold.
    profiling: u64,
}

/// An expert redistribution between its pairing and its commit.
enum Experts {
    /// The weights are paired, and the copies wait to start.
    Paired(Replacement),
    /// The copies run beside the serving batches.
    Copying(std::thread::JoinHandle<Result<Replacement>>),
    /// The copies completed, and the new weights wait for the common boundary.
    Ready(Replacement),
}

/// What a rank holds of a loaded LoRA adapter: its tensors, published and mapped read-only.
struct Adapter {
    /// The store that maps the adapter's tensors, which lives as long as they are lent.
    store: WeightStore,
    /// The adapter's weights, which an unload ends the retention of.
    keys: Vec<WeightKey>,
    /// Each tensor by its name in the adapter's file. Nothing reads them until a kernel package
    /// of the epoch does; a slot's tensors are what such a package is handed.
    tensors: BTreeMap<String, nn::Tensor>,
}

/// The weight binding of one process: every rank it holds.
pub(crate) struct WeightBinding {
    ranks: Vec<Mutex<Rank>>,
    /// The links to the epoch's other main processes, each with the worker groups it holds ranks
    /// of, in the order of their first ranks; none on a peer and on an epoch of one process.
    remotes: Vec<(Vec<GroupId>, Arc<Remote>)>,
    /// What the ranks serve now.
    target: Mutex<Target>,
    /// A weight update's load, waiting for its boundary.
    staged: Mutex<Option<Replacement>>,
    experts: Mutex<Option<Experts>>,
    /// Each rank's adapters, by the slot the daemon assigned.
    adapters: Mutex<BTreeMap<u32, Vec<Adapter>>>,
    /// Whether the graph pools are suspended, which only a restore reverses.
    suspended: AtomicBool,
    /// The weights changed while the graph pools were suspended: what the runners derive from
    /// them is computed in the pools, so the restore recomputes it.
    derived_stale: AtomicBool,
    /// A mapping change failed part of the way: the rank's weights are no one checkpoint's, and
    /// the epoch ends rather than computing with them.
    poisoned: Mutex<Option<String>>,
    /// What each rank's process reserves, in rank order, once the ranks serve.
    reserved: Mutex<Vec<Reservation>>,
}

fn rejected(what: &str, error: impl std::fmt::Display) -> Error {
    Error::Invalid(format!("{what}: {error}"))
}

/// The digest a set of weights is known by, from their load's order: what roots the prefix hash
/// chains of the pages computed with them, so a page of another checkpoint never matches.
pub(crate) fn identity_of(order: &[(WeightKey, bool)]) -> String {
    identity_of_names(order.iter().map(|(key, _)| key.to_string()))
}

/// [`identity_of`] the weights of `names`, each weight's name (its key's text) in their load's
/// order.
pub(crate) fn identity_of_names(names: impl IntoIterator<Item = impl AsRef<str>>) -> String {
    let mut digest = ::storage::key::Digest::new(b"uniserve-weights-identity");
    for name in names {
        digest.text(name.as_ref());
    }
    ::storage::Key(digest.finish()).hex()
}

/// What one rank serves, as the identities of the deployment's weights are computed from it: its
/// worker group, whether it computes encoder outputs, and the names of its weights in its load's
/// order.
pub(crate) type Served = (GroupId, bool, Vec<String>);

/// What `rank`, of worker group `group`, serves: its weights' names in their load's order.
pub(crate) fn served(group: GroupId, encodes: bool, weights: &[(WeightKey, bool)]) -> Served {
    let names = weights.iter().map(|(key, _)| key.to_string()).collect();
    (group, encodes, names)
}

/// The identity of the weights that compute each worker group's encoder outputs
/// (`Capacity::encoder_identity`), from `ranks`, what every rank of the deployment serves in rank
/// order: per group, in the order of its first encoding rank, the identity of its encoding ranks'
/// weights in rank order ([`identity_of`]). The rank start states it, and a weight update renews
/// it from the same ranks, so one set of weights keys its outputs alike in both.
pub(crate) fn encoder_identities(ranks: &[Served]) -> Vec<(GroupId, String)> {
    let mut groups: Vec<(GroupId, Vec<&str>)> = Vec::new();
    for (group, _, names) in ranks.iter().filter(|(_, encodes, _)| *encodes) {
        let names = names.iter().map(String::as_str);
        match groups.iter_mut().find(|(id, _)| id == group) {
            Some((_, held)) => held.extend(names),
            None => groups.push((*group, names.collect())),
        }
    }

    groups
        .into_iter()
        .map(|(group, names)| (group, identity_of_names(names)))
        .collect()
}

/// The identity the pages of every rank of the deployment are keyed by: of the weights of
/// `ranks`, every rank's in rank order ([`identity_of`]).
pub(crate) fn page_identity(ranks: &[Served]) -> String {
    identity_of_names(ranks.iter().flat_map(|(_, _, names)| names))
}

/// Refuses `tables`, every rank's segment table of the regions an operation created segments
/// of, when two ranks hold different tables: the head grants a segment once every rank that
/// maps its region holds it, which only tables that agree state.
fn agreed(tables: &[Option<CacheTable>]) -> Result<()> {
    let mut held = tables.iter().flatten();
    let Some(first) = held.next() else {
        return Ok(());
    };
    match held.find(|table| *table != first) {
        Some(other) => Err(Error::Invariant(format!(
            "the ranks of one region hold different segment tables: {first:?} and {other:?}"
        ))),
        None => Ok(()),
    }
}

/// Each peer's value of an operation, decoded from the JSON it answered, in the order of
/// `answers`, once every one answered.
///
/// # Errors
///
/// The first peer's failure, or a reply that is not the value's JSON.
fn answered<T: DeserializeOwned>(answers: Vec<(u32, Answer)>) -> Result<Vec<T>> {
    let mut values = Vec::with_capacity(answers.len());
    let mut failure = None;
    for (rank, answer) in answers {
        let value = operated(rank, &answer).and_then(|reply| {
            serde_json::from_str(&reply).map_err(|error| {
                Error::Invalid(format!(
                    "rank {rank}'s engine process answered an operation with {reply:?}: {error}"
                ))
            })
        });
        match value {
            Ok(value) => values.push(value),
            Err(error) => {
                failure.get_or_insert(error);
            }
        }
    }

    match failure {
        Some(error) => Err(error),
        None => Ok(values),
    }
}

/// `here`, this process's part of an operation, and `there`, the peers', once both ended: the
/// first failure, or both values.
fn both<T, U>(here: Result<T>, there: Result<U>) -> Result<(T, U)> {
    Ok((here?, there?))
}

/// Every rank's segment table, this process's (`here`) then each peer's (`answers`), in rank
/// order.
fn joined_tables(
    here: Result<Vec<Option<CacheTable>>>,
    answers: Vec<(u32, Answer)>,
) -> Result<Vec<Option<CacheTable>>> {
    let (mut tables, peers) = both(here, answered::<Vec<Option<CacheTable>>>(answers))?;
    tables.extend(peers.into_iter().flatten());
    Ok(tables)
}

/// The room `rank`'s device has ([`Room::of`]).
fn room_of(rank: &Rank) -> Result<Room> {
    let (quota, _) = rank.client.usage(rank.device)?;
    let free = rank.device.memory_info()?.free;

    Ok(Room::of(&quota, free, rank.client.epoch()))
}

/// The pools of a process's ranks on one device, and the bytes of those ranks' communicators that
/// NVML attributes to no process.
type DevicePools = (Vec<Arc<Mutex<GraphPool>>>, u64);

/// The bytes this process holds on `device` beside `pools`, those of every rank of the process
/// there, and the Storage Server's storages, with `unattributed` added
/// ([`process_bytes`]). Each pool is locked only while its mapped bytes are read.
pub(crate) fn measure(
    nvml: &Nvml,
    device: DeviceId,
    pools: &[Arc<Mutex<GraphPool>>],
    unattributed: u64,
) -> Result<u64> {
    let mapped = pools.iter().map(|pool| lock(pool).mapped()).sum();
    process_bytes(nvml, device, mapped, unattributed)
}

/// The segments of `table` a warm rank keeps: those it holds from the first, less the fewest at
/// the tail whose bytes (`unit_bytes` a unit) cover `excess`, the bytes by which its GPU's quota
/// would pass its limit once the rank reserves its warm process (`division::excess`). Only the
/// segments epoch `epoch` created are given back: a segment another epoch created and the rank
/// maps stays that epoch's storage, and giving it back would free nothing of the quota.
fn kept_beside(table: &CacheTable, unit_bytes: u64, epoch: u64, excess: u64) -> u32 {
    let mut kept = table.held;
    let mut freed: u64 = 0;
    while freed < excess && kept > 0 {
        let Some(segment) = table.segments.get(kept as usize - 1) else {
            break;
        };
        if segment.epoch != epoch {
            break;
        }
        freed = freed.saturating_add(u64::from(segment.units).saturating_mul(unit_bytes));
        kept -= 1;
    }
    kept
}

/// What `cache`'s memory of the device region `region` is made of and how many of its segments
/// exist: its KV planes or its state units (`KvCacheMemory::segment_table`); none for memory that
/// is not split or a region that is none of the cache's.
fn segment_table(cache: &KvCacheMemory, region: u8) -> Option<CacheTable> {
    cache.segment_table(region).map(|(table, _)| table)
}

/// One rank's second load of `target` through `weights`, beside the weights of `serving`: it
/// publishes what changed and returns the load's order. A load that fails leaves the store with
/// no load in progress; what it published is the discard's to release.
fn reload(
    model: &dyn ModelLoad,
    weights: &WeightStore,
    serving: &[(WeightKey, bool)],
    target: &Target,
) -> Result<Vec<(WeightKey, bool)>> {
    let load = Reload::new(weights, serving)?;
    let loaded = match model.load(target, &load) {
        Ok(()) => load.finish(),
        Err(refusal) => Err(Error::Invalid(refusal)),
    };

    if loaded.is_err() {
        weights.abandon();
    }
    loaded
}

impl WeightBinding {
    /// The weight binding over `ranks`, which serve `target`, and the ranks of the peers behind
    /// `remotes`, the links to the epoch's other main processes with the worker groups each holds
    /// ranks of, in the order of their first ranks.
    pub(crate) fn new(
        ranks: Vec<Rank>,
        target: Target,
        remotes: Vec<(Vec<GroupId>, Arc<Remote>)>,
    ) -> Self {
        Self {
            ranks: ranks.into_iter().map(Mutex::new).collect(),
            remotes,
            target: Mutex::new(target),
            staged: Mutex::new(None),
            experts: Mutex::new(None),
            adapters: Mutex::new(BTreeMap::new()),
            suspended: AtomicBool::new(false),
            derived_stale: AtomicBool::new(false),
            poisoned: Mutex::new(None),
            reserved: Mutex::new(Vec::new()),
        }
    }

    /// Runs the operation `work` makes for each rank, given its index, as a control operation
    /// named `label` on the rank's worker, and returns each rank's value in rank order.
    ///
    /// # Errors
    ///
    /// As [`Self::each_of`].
    fn each<T: Send + 'static>(
        &self,
        label: &str,
        work: impl Fn(usize, &Rank) -> Work<T>,
    ) -> Result<Vec<T>> {
        self.each_of(None, label, work)
    }

    /// Runs the operation `work` makes for each rank of the worker groups `groups` (every rank
    /// for `None`), given its index among all ranks, as a control operation named `label` on the
    /// rank's worker, and returns each such rank's value in rank order.
    ///
    /// # Errors
    ///
    /// The first rank's failure, once every rank's operation has ended: a rank whose worker
    /// closed, an operation that failed, or a worker that closed before it ran the operation.
    fn each_of<T: Send + 'static>(
        &self,
        groups: Option<&[GroupId]>,
        label: &str,
        work: impl Fn(usize, &Rank) -> Work<T>,
    ) -> Result<Vec<T>> {
        // Every rank's operation is handed over before any is waited for.
        let mut submitted = Vec::with_capacity(self.ranks.len());
        for (index, rank) in self.ranks.iter().enumerate() {
            let rank = lock(rank);
            if let Some(groups) = groups
                && !rank.of(groups)
            {
                continue;
            }
            let work = work(index, &rank);
            let (value, produced) = crossbeam_channel::bounded(1);

            let submission = rank.worker().and_then(|worker| {
                worker.control(
                    format!("{label} on rank {}", rank.rank),
                    Box::new(move |runners: &mut [Box<dyn Runner>]| {
                        // A weight binding that stopped waiting has nothing to be told.
                        let _ = value.send(work(runners)?);
                        Ok(())
                    }),
                )
            });
            submitted.push((submission, produced));
        }

        // A failure is reported once every operation has ended, so the caller acts on it with
        // nothing of the change still running on any device.
        let mut values = Vec::with_capacity(submitted.len());
        let mut failure = None;
        for (submission, produced) in submitted {
            let ended = submission.and_then(|submission| match submission.wait() {
                Outcome::Success(_) => produced
                    .recv()
                    .map_err(|_| Error::State("a control operation ended without its value")),
                Outcome::Failed(fault) => Err(Error::Shared(fault)),
                Outcome::Cancelled => Err(Error::State(
                    "the rank's worker closed before it ran the operation",
                )),
            });

            match ended {
                Ok(value) => values.push(value),
                Err(error) => {
                    failure.get_or_insert(error);
                }
            }
        }

        match failure {
            Some(error) => Err(error),
            None => Ok(values),
        }
    }

    /// Sends `operation` to every peer link that holds a rank of `groups`, or to every link for
    /// `None`, before anything of this process's part is waited for, and returns where each
    /// answers with the link's first rank, in link order.
    ///
    /// # Errors
    ///
    /// A link that is lost.
    fn ask_peers(
        &self,
        groups: Option<&[GroupId]>,
        operation: impl Fn() -> DataOperation,
    ) -> Result<Vec<(u32, Answer)>> {
        self.remotes
            .iter()
            .filter(|(held, _)| {
                groups.is_none_or(|groups| held.iter().any(|group| groups.contains(group)))
            })
            .map(|(_, remote)| Ok((remote.rank(), remote.operate(operation())?)))
            .collect()
    }

    /// Runs the operation the daemon names `name` with the JSON `payload` on every peer's ranks
    /// (`crate::process::operations::hook` runs it on this process's), and returns each peer
    /// rank's result, in rank order.
    ///
    /// # Errors
    ///
    /// The first peer's failure.
    pub(crate) fn hook_peers(&self, name: &str, payload: &str) -> Result<Vec<String>> {
        let answers = self.ask_peers(None, || DataOperation::Hook {
            name: name.to_owned(),
            payload: payload.to_owned(),
        })?;
        let results: Vec<Vec<String>> = answered(answers)?;
        Ok(results.into_iter().flatten().collect())
    }

    /// Reserves what each rank's process holds beside its pools and the Storage Server's storages
    /// in its device's quota, as the ranks start to serve, the peers' ranks in their processes;
    /// [`Self::reserve_here`] is this process's part.
    ///
    /// # Errors
    ///
    /// As [`Self::reserve_here`], or a peer's failure.
    pub(crate) fn reserve_serving(&self) -> Result<()> {
        let answers = self.ask_peers(None, || DataOperation::ReserveServing)?;
        let here = self.reserve_here();
        both(here, answered::<()>(answers)).map(drop)
    }

    /// The pools of this process's ranks on each of their devices, with the bytes of those
    /// ranks' communicators that NVML attributes to no process: what a measurement of the
    /// process on the device takes out of NVML's figure and adds to it ([`measure`]).
    fn device_pools(&self) -> BTreeMap<DeviceId, DevicePools> {
        let mut devices: BTreeMap<DeviceId, DevicePools> = BTreeMap::new();
        for rank in &self.ranks {
            let rank = lock(rank);
            let (pools, unattributed) = devices.entry(rank.device).or_default();
            pools.push(Arc::clone(&rank.graphs));
            *unattributed += rank.unattributed;
        }
        devices
    }

    /// Reserves what the process holds on each of its ranks' devices beside their pools and the
    /// Storage Server's storages in the device's quota, as the ranks start to serve: its CUDA
    /// context, its kernels and its communicators, measured once the worker has passed every job
    /// before it. The process's first rank on a device reserves the figure and its other ranks
    /// there none, since NVML states the process whole. From here on the quota binds the whole
    /// rank, so a storage that would take the deployment past it is refused at the Storage Server
    /// and not found out by the device.
    ///
    /// Reserved once: a process that already reserved keeps its figures, and a standby reserves
    /// when it is promoted, because until then the quota counts what it maps of its predecessor.
    ///
    /// # Errors
    ///
    /// When a rank's process does not fit its quota, or NVML cannot state the process's storage.
    fn reserve_here(&self) -> Result<()> {
        let mut reserved = lock(&self.reserved);
        // A weight binding of no rank has no process storage to measure.
        if !reserved.is_empty() || self.ranks.is_empty() {
            return Ok(());
        }

        let nvml = Arc::new(Nvml::init()?);
        let devices = self.device_pools();
        let measured = self.each("measuring the process's device storage", |_, rank| {
            let nvml = Arc::clone(&nvml);
            let device = rank.device;
            let (pools, unattributed) = devices.get(&device).cloned().unwrap_or_default();

            Box::new(move |_: &mut [Box<dyn Runner>]| measure(&nvml, device, &pools, unattributed))
        })?;

        // The figure is the process's whole on the device: the first of its ranks there reserves
        // it, and the others none.
        let mut counted = BTreeSet::new();
        let reservations: Vec<Reservation> = self
            .ranks
            .iter()
            .zip(measured)
            .map(|(rank, process)| Reservation {
                process: if counted.insert(lock(rank).device) {
                    process
                } else {
                    0
                },
                profiling: 0,
            })
            .collect();
        for (rank, reservation) in self.ranks.iter().zip(&reservations) {
            let rank = lock(rank);
            tracing::info!(
                device = rank.device.0,
                process = reservation.process,
                "the serving rank reserves its process's device storage"
            );
            self.reserve(&rank, reservation)?;
        }

        *reserved = reservations;
        Ok(())
    }

    /// What each of this process's ranks reserved as its warm process in its GPU's quota
    /// ([`Self::reserve_serving`]), with the rank's device; none before the ranks reserved.
    pub(crate) fn warm_processes(&self) -> Vec<(DeviceId, u64)> {
        let reserved = lock(&self.reserved);
        self.ranks
            .iter()
            .zip(reserved.iter())
            .map(|(rank, reservation)| (lock(rank).device, reservation.process))
            .collect()
    }

    /// States `reservation` as `rank`'s process in its GPU's quota.
    fn reserve(&self, rank: &Rank, reservation: &Reservation) -> Result<()> {
        let total = rank.device.memory_info()?.total;
        let bytes = reservation.process.saturating_add(reservation.profiling);

        rank.client.reserve(
            rank.device,
            total,
            quota_limit(total, rank.fraction),
            vec![(Item::Process, bytes)],
        )?;
        Ok(())
    }

    /// Reserves `bytes` of device storage a session's collectors hold on `index`'s device with
    /// the rank's process. Returns the quota's refusal; a process that reserves nothing yet, a
    /// standby, has nothing to reserve it beside.
    fn reserve_profiling(&self, index: usize, bytes: u64) -> Result<Option<String>> {
        let mut reserved = lock(&self.reserved);
        let (Some(reservation), Some(rank)) = (reserved.get_mut(index), self.ranks.get(index))
        else {
            return Ok(None);
        };
        if reservation.profiling == bytes {
            return Ok(None);
        }

        let held = std::mem::replace(&mut reservation.profiling, bytes);
        match self.reserve(&lock(rank), reservation) {
            Ok(()) => Ok(None),
            Err(error) => {
                reservation.profiling = held;
                match error {
                    // `StorageClient::reserve` states the server's refusal of a reservation past
                    // the quota so.
                    Error::Unsupported(refusal) => Ok(Some(refusal)),
                    error => Err(error),
                }
            }
        }
    }

    /// Starts the profiling session `payload` states on every rank (`docs/profiling.md`): each
    /// reserves what its collectors hold, then starts at its next batch boundary. A rank whose
    /// quota cannot fit the storage responds that it refused, and the daemon ends what the others
    /// started.
    ///
    /// # Errors
    ///
    /// When the payload is no session, or a rank cannot create its file.
    pub(crate) fn start_profile(&self, payload: &str) -> Result<Vec<String>> {
        let start: proto::profile::SessionStart =
            serde_json::from_str(payload).map_err(|error| rejected("profile_start", error))?;

        let mut responses = Vec::with_capacity(self.ranks.len());
        for (index, rank) in self.ranks.iter().enumerate() {
            let (number, worker) = {
                let rank = lock(rank);
                (rank.rank, rank.worker()?)
            };

            let held = crate::profiling::session::device_storage(&start.request, number);
            if let Some(refusal) = self.reserve_profiling(index, held)? {
                responses.push(
                    serde_json::json!({
                        "rank": number,
                        "selected": true,
                        "refused": format!(
                            "range_metrics holds {held} bytes of device storage for its range \
                             profiler pass, which the quota of the rank's GPU cannot fit: \
                             {refusal}"
                        ),
                    })
                    .to_string(),
                );
                continue;
            }

            let started = worker.profile(Change::Start {
                start: Box::new(start.clone()),
                rank: number,
            });
            match started {
                Ok(response) => responses.push(response.to_string()),
                Err(error) => {
                    self.release_profiling(index);
                    return Err(error);
                }
            }
        }
        Ok(responses)
    }

    /// The session's storage is no longer held on `index`'s device. A failure leaves the larger
    /// reservation in the quota, which only overstates the rank's storage, so it is logged rather
    /// than failing the session's end.
    fn release_profiling(&self, index: usize) {
        let failure = match self.reserve_profiling(index, 0) {
            Ok(None) => return,
            Ok(Some(refusal)) => refusal,
            Err(error) => error.to_string(),
        };
        tracing::warn!("releasing the ended profiling session's device storage failed: {failure}");
    }

    /// Ends the profiling session `payload` names on every rank: each writes what remains and
    /// closes its file, and the quota gives back what the session held.
    ///
    /// # Errors
    ///
    /// When the payload names no session, or a rank's worker is gone.
    pub(crate) fn stop_profile(&self, payload: &str) -> Result<Vec<String>> {
        let value: serde_json::Value =
            serde_json::from_str(payload).map_err(|error| rejected("profile_stop", error))?;
        let session = value["session"]
            .as_u64()
            .ok_or_else(|| Error::Invalid("profile_stop names no session".into()))?;
        let reason = value["reason"].as_str().unwrap_or("stopped").to_owned();

        let mut responses = Vec::with_capacity(self.ranks.len());
        for (index, rank) in self.ranks.iter().enumerate() {
            let (number, worker) = {
                let rank = lock(rank);
                (rank.rank, rank.worker()?)
            };

            let response = worker.profile(Change::Stop {
                session,
                reason: reason.clone(),
                rank: number,
            })?;
            self.release_profiling(index);
            responses.push(response.to_string());
        }
        Ok(responses)
    }

    /// Turns one collector of a session off on every rank, for the quality-of-service guard.
    ///
    /// # Errors
    ///
    /// When the payload names no session or collector, or a rank's worker is gone.
    pub(crate) fn release_profile(&self, payload: &str) -> Result<Vec<String>> {
        let value: serde_json::Value =
            serde_json::from_str(payload).map_err(|error| rejected("profile_release", error))?;
        let session = value["session"]
            .as_u64()
            .ok_or_else(|| Error::Invalid("profile_release names no session".into()))?;
        let collector: proto::profile::Collector =
            serde_json::from_value(value["collector"].clone())
                .map_err(|error| rejected("profile_release", error))?;
        let reason = value["reason"].as_str().unwrap_or("released").to_owned();

        let mut responses = Vec::with_capacity(self.ranks.len());
        for rank in &self.ranks {
            let (number, worker) = {
                let rank = lock(rank);
                (rank.rank, rank.worker()?)
            };

            let response = worker.profile(Change::Release {
                session,
                collector,
                reason: reason.clone(),
                rank: number,
            })?;
            responses.push(response.to_string());
        }
        Ok(responses)
    }

    /// Why this epoch must end, once a change of its mappings failed part of the way.
    pub(crate) fn poisoned(&self) -> Option<String> {
        lock(&self.poisoned).clone()
    }

    /// The target the ranks serve now.
    pub(crate) fn target(&self) -> Target {
        lock(&self.target).clone()
    }

    /// What every rank of the deployment this binding reaches serves, in rank order: this
    /// process's ranks, then each peer's. What a rank's pages and encoder outputs hold follows
    /// from all of them: an expert union's rank computes each token's experts on its peers, and a
    /// tensor rank's activations reach the others', so a page is another page once any rank's
    /// weights change, and every rank keys it alike.
    ///
    /// # Errors
    ///
    /// A peer's failure.
    pub(crate) fn served(&self) -> Result<Vec<Served>> {
        let mut ranks = self.served_here();
        ranks.extend(self.served_by_peers()?);
        Ok(ranks)
    }

    /// What every peer's rank serves, in rank order; none on an epoch of one process.
    ///
    /// # Errors
    ///
    /// A peer's failure.
    pub(crate) fn served_by_peers(&self) -> Result<Vec<Served>> {
        let answers = self.ask_peers(None, || DataOperation::ServingOrder)?;
        Ok(answered::<Vec<Served>>(answers)?
            .into_iter()
            .flatten()
            .collect())
    }

    /// Waits until every peer's own preparation of generation `generation` ended, which its
    /// daemon commands (`docs/architecture.md`, "Generations"): no shadow batch of the generation
    /// reaches a peer's ranks before they hold it.
    ///
    /// # Errors
    ///
    /// A peer whose preparation failed, or did not end within its bound.
    pub(crate) fn prepared(&self, generation: u64) -> Result<()> {
        let answers = self.ask_peers(None, || DataOperation::Prepared { generation })?;
        answered::<()>(answers).map(drop)
    }

    /// The operators every peer's ranks have no comparison for in the generation update in
    /// flight, in rank order: those their preparation added, and those the shadow batches that
    /// reached them could not compare, so the caller asks once its shadow pass ended. None on an
    /// epoch of one process.
    ///
    /// # Errors
    ///
    /// A peer's failure.
    pub(crate) fn uncompared_by_peers(&self) -> Result<Vec<String>> {
        let answers = self.ask_peers(None, || DataOperation::Uncompared)?;
        Ok(answered::<Vec<String>>(answers)?
            .into_iter()
            .flatten()
            .collect())
    }

    /// What this process's ranks serve, in rank order; a rank whose worker closed serves none.
    pub(crate) fn served_here(&self) -> Vec<Served> {
        self.ranks
            .iter()
            .filter_map(|rank| {
                let rank = lock(rank);
                let group = rank.worker.upgrade()?.group();
                Some(served(group, rank.encodes, &rank.serving))
            })
            .collect()
    }

    /// The identity of the weights that compute each worker group's encoder outputs, as every
    /// rank of the deployment serves them now ([`encoder_identities`]): what a weight update keys
    /// the outputs computed after it by. A group none of whose ranks encodes has none.
    ///
    /// # Errors
    ///
    /// A peer's failure.
    pub(crate) fn encoder_identities(&self) -> Result<Vec<(GroupId, String)>> {
        Ok(encoder_identities(&self.served()?))
    }

    /// The room every device has for a load, in rank order, the peers' ranks after this
    /// process's.
    ///
    /// # Errors
    ///
    /// When the quota or the device cannot be read, or a peer's failure.
    pub(crate) fn room(&self) -> Result<Vec<Room>> {
        let answers = self.ask_peers(None, || DataOperation::Room)?;
        let here: Result<Vec<Room>> = self.ranks.iter().map(|rank| room_of(&lock(rank))).collect();
        let (mut rooms, peers) = both(here, answered::<Vec<Room>>(answers))?;
        rooms.extend(peers.into_iter().flatten());
        Ok(rooms)
    }

    /// What the ranks' devices map now, one entry per device in the order of its first rank, as
    /// the process's metrics report it: the limit of the GPU's quota the mappings are reserved
    /// against, the bytes each pool of the device's allocators maps (`GraphPool::usage`), and
    /// the bytes the planes of the ranks' KV regions map, as the pool [`KV_POOL`]. The ranks of
    /// one device add up.
    ///
    /// A rank whose owners another thread holds at the moment is left out of this sample: the
    /// control thread that takes it once a second waits on no rank, and a series a report does
    /// not name keeps its last value.
    pub(crate) fn memory(&self) -> Vec<DeviceMemory> {
        let mut devices: Vec<DeviceMemory> = Vec::new();
        for rank in &self.ranks {
            let Ok(rank) = rank.try_lock() else {
                continue;
            };
            let Ok(graphs) = rank.graphs.try_lock() else {
                continue;
            };
            let mut stated = graphs.usage();
            drop(graphs);
            if let Some(cache) = &rank.cache
                && let Ok(cache) = cache.try_lock()
            {
                stated.pools.push(PoolMemory {
                    pool: KV_POOL.to_owned(),
                    mapped: cache.mapped_bytes(),
                });
            }

            match devices
                .iter_mut()
                .find(|device| device.device == stated.device)
            {
                Some(device) => {
                    for pool in stated.pools {
                        match device.pools.iter_mut().find(|held| held.pool == pool.pool) {
                            Some(held) => held.mapped += pool.mapped,
                            None => device.pools.push(pool),
                        }
                    }
                }
                None => devices.push(stated),
            }
        }
        devices
    }

    /// The bytes of device storage the graph pools map, on the device that maps the fewest over
    /// every rank, the peers' included: what a suspension of the pools gives back on every
    /// device. Zero for a binding of no rank.
    ///
    /// # Errors
    ///
    /// A peer's failure.
    pub(crate) fn graph_bytes(&self) -> Result<u64> {
        let answers = self.ask_peers(None, || DataOperation::GraphBytes)?;
        let here = self.graph_bytes_here();
        let peers: Vec<Option<u64>> = answered(answers)?;
        Ok(peers.into_iter().chain([here]).flatten().min().unwrap_or(0))
    }

    /// [`Self::graph_bytes`] over this process's ranks; none for a process of no rank.
    pub(crate) fn graph_bytes_here(&self) -> Option<u64> {
        self.ranks
            .iter()
            .map(|rank| Arc::clone(&lock(rank).graphs))
            .map(|graphs| lock(&graphs).mapped())
            .min()
    }

    /// Pairs the weights of a load of `target` with the ones that serve, on every rank, the
    /// peers' included, each of which keeps its own share ([`Self::pair_here`]).
    ///
    /// # Errors
    ///
    /// As [`Self::pair_here`], on any rank.
    pub(crate) fn pair(&self, target: Target, rearranged: bool) -> Result<Replacement> {
        let checkpoint = target.checkpoint.to_string_lossy().into_owned();
        let answers = self.ask_peers(None, || DataOperation::Pair {
            checkpoint: checkpoint.clone(),
            experts: target.experts.clone(),
            rearranged,
        })?;
        let here = self.pair_here(target, rearranged);
        let (mut replacement, peers) = both(here, answered::<u64>(answers))?;

        replacement.peers = peers.into_iter().max().unwrap_or(0);
        Ok(replacement)
    }

    /// Pairs the weights of a load of `target` with the ones that serve, on every rank of this
    /// process, moving no byte: the model's load runs over storage on the meta device, which
    /// states the weights it reaches without reading a payload. With `rearranged`, a stack may
    /// hold the same parts in another order (an expert redistribution); without, each weight
    /// reads the same tensors (a weight update).
    ///
    /// # Errors
    ///
    /// When the target cannot be the served model: the sentence names its first difference.
    pub(crate) fn pair_here(&self, target: Target, rearranged: bool) -> Result<Replacement> {
        let mut ranks = Vec::with_capacity(self.ranks.len());
        for rank in &self.ranks {
            let (model, weights, serving) = {
                let rank = lock(rank);
                (
                    Arc::clone(&rank.model),
                    Arc::clone(&rank.weights),
                    rank.serving.clone(),
                )
            };

            let meta = MetaWeights::new();
            model.load(&target, &meta).map_err(Error::Invalid)?;
            let order = meta.into_order();

            let pairs = paired(&serving, &order, rearranged).map_err(|difference| {
                Error::Invalid(format!(
                    "{} loads other tensors than the served model: {difference}",
                    target.checkpoint.display()
                ))
            })?;

            // A replacement of a weight of its own takes an allocation of its length, and one of
            // a packed weight its share of an allocation.
            let mut bytes: u64 = 0;
            for (serving, _) in &pairs {
                bytes = bytes.saturating_add(weights.replacement_bytes(serving)?);
            }
            ranks.push(Share {
                pairs,
                order,
                bytes,
            });
        }

        Ok(Replacement {
            target,
            ranks,
            peers: 0,
            loaded: false,
        })
    }

    /// Publishes what `replacement` changes on every rank, each peer its own share
    /// ([`Self::load_here`]).
    ///
    /// # Errors
    ///
    /// When a rank's load fails: everything this load published is released on every rank, and
    /// the ranks serve on as they did.
    pub(crate) fn load(&self, replacement: Replacement) -> Result<Replacement> {
        let answers = self.ask_peers(None, || DataOperation::Load)?;
        let here = self.load_here(replacement);
        let there = answered::<()>(answers);

        match (here, there) {
            (Ok(replacement), Ok(_)) => Ok(replacement),
            (Ok(replacement), Err(error)) => {
                self.discard(replacement);
                Err(error)
            }
            // This process's share is released already; the peers' goes too.
            (Err(error), _) => {
                self.discard_peers();
                Err(error)
            }
        }
    }

    /// Publishes what `replacement` changes on every rank of this process, each rank on a thread
    /// of its own: the batches go on reading the weights that serve.
    ///
    /// # Errors
    ///
    /// When a rank's load fails: everything this load published is released, and the ranks serve
    /// on as they did.
    pub(crate) fn load_here(&self, mut replacement: Replacement) -> Result<Replacement> {
        let target = replacement.target.clone();

        // One rank's share of the load, copied out so no rank's lock is held while the loads run.
        type Inputs = (Arc<dyn ModelLoad>, Arc<WeightStore>, Vec<(WeightKey, bool)>);
        let inputs: Vec<Inputs> = self
            .ranks
            .iter()
            .map(|rank| {
                let rank = lock(rank);
                (
                    Arc::clone(&rank.model),
                    Arc::clone(&rank.weights),
                    rank.serving.clone(),
                )
            })
            .collect();
        replacement.loaded = true;

        let orders: Vec<Result<Vec<(WeightKey, bool)>>> = std::thread::scope(|scope| {
            let handles: Vec<_> = inputs
                .iter()
                .map(|(model, weights, serving)| {
                    let target = &target;
                    scope.spawn(move || reload(model.as_ref(), weights, serving, target))
                })
                .collect();

            handles
                .into_iter()
                .map(|handle| {
                    handle.join().unwrap_or_else(|_| {
                        Err(Error::Invariant("a rank's weight load panicked".into()))
                    })
                })
                .collect()
        });

        let mut failure = None;
        for (share, order) in replacement.ranks.iter().zip(orders) {
            match order {
                // The pairing and the load are one code path over one checkpoint, so a difference
                // here is a checkpoint that changed between them.
                Ok(order) if order == share.order => {}
                Ok(_) => {
                    failure.get_or_insert_with(|| {
                        format!(
                            "{} changed while it loaded: its weights are not the ones paired",
                            target.checkpoint.display()
                        )
                    });
                }
                Err(error) => {
                    failure.get_or_insert(error.to_string());
                }
            }
        }

        if let Some(failure) = failure {
            self.discard_here(replacement);
            return Err(Error::Invalid(format!(
                "the weights of {} did not load: {failure}",
                target.checkpoint.display()
            )));
        }
        Ok(replacement)
    }

    /// Ends a replacement that will not serve, on every rank: what its load published goes back,
    /// and the ranks serve on.
    pub(crate) fn discard(&self, replacement: Replacement) {
        self.discard_here(replacement);
        self.discard_peers();
    }

    /// Ends each peer's share of the replacement that will not serve. A failure leaves storage
    /// published that nothing reads, which only overstates the peer's storage, so it is logged.
    fn discard_peers(&self) {
        let discarded = self
            .ask_peers(None, || DataOperation::Discard)
            .and_then(answered::<()>);
        if let Err(error) = discarded {
            tracing::warn!(%error, "a peer's abandoned weight load was not released");
        }
    }

    /// Ends a replacement of this process's ranks that will not serve: what its load published
    /// goes back, and the ranks serve on.
    pub(crate) fn discard_here(&self, replacement: Replacement) {
        if !replacement.loaded {
            return;
        }

        for (rank, share) in self.ranks.iter().zip(&replacement.ranks) {
            let weights = Arc::clone(&lock(rank).weights);
            let published: Vec<WeightKey> =
                share.pairs.iter().map(|(_, new)| new.clone()).collect();

            if let Err(error) = weights.forget(&published) {
                tracing::warn!(%error, "an abandoned weight load's storage was not released");
            }
        }
    }

    /// Keeps a loaded replacement for the boundary that puts it in place
    /// ([`Self::update_weights`]), each peer its own share.
    ///
    /// # Errors
    ///
    /// A peer that holds no loaded share.
    pub(crate) fn stage(&self, replacement: Replacement) -> Result<()> {
        let answers = self.ask_peers(None, || DataOperation::Stage)?;
        self.stage_here(replacement);
        answered::<()>(answers).map(drop)
    }

    /// Keeps a loaded replacement of this process's ranks for the boundary that puts it in place.
    pub(crate) fn stage_here(&self, replacement: Replacement) {
        if let Some(previous) = lock(&self.staged).replace(replacement) {
            self.discard_here(previous);
        }
    }

    /// Puts the staged replacement in place on every rank, the peers' included, and returns each
    /// rank's report, in rank order.
    ///
    /// The caller has paused every scheduler whose batches read the ranks' weights, so every
    /// rank of a group changes between the same two batches.
    ///
    /// # Errors
    ///
    /// When no replacement is staged, and as [`Self::commit_experts`].
    pub(crate) fn update_weights(&self) -> Result<Vec<WeightsUpdated>> {
        let answers = self.ask_peers(None, || DataOperation::UpdateWeights)?;
        let here = lock(&self.staged)
            .take()
            .ok_or(Error::State(
                "no weights are loaded for this update to put in place",
            ))
            .and_then(|replacement| self.rebind(replacement));
        self.reported(here, answers)
    }

    /// The reports of a change of every rank's weights: this process's (`here`) and each peer's
    /// (`answers`), in rank order. A peer that changed part of the way leaves the deployment's
    /// weights no one checkpoint's, which poisons this binding as a rank of its own would.
    ///
    /// Every rank keys its pages by the weights of every rank of the deployment
    /// ([`Self::served`]), which only this process reaches: each report that states page
    /// identities states the ones this process's ranks give for all of them, every rank of a
    /// text deployment laying its cache out as the others do.
    fn reported(
        &self,
        here: Result<Vec<WeightsUpdated>>,
        answers: Vec<(u32, Answer)>,
    ) -> Result<Vec<WeightsUpdated>> {
        let there = answered::<Vec<WeightsUpdated>>(answers);
        if let Err(error) = &there {
            lock(&self.poisoned).get_or_insert_with(|| error.to_string());
        }
        let (mut reports, peers) = both(here, there)?;
        reports.extend(peers.into_iter().flatten());
        if self.remotes.is_empty() {
            return Ok(reports);
        }

        let fingerprint = page_identity(&self.served()?);
        let identities = self.ranks.iter().find_map(|rank| {
            let rank = lock(rank);
            let cache = rank.cache.as_ref()?;
            Some(lock(cache).page_identities(&fingerprint))
        });
        if let Some(identities) = identities {
            for report in reports
                .iter_mut()
                .filter(|report| !report.identities.is_empty())
            {
                report.identities.clone_from(&identities);
            }
        }
        Ok(reports)
    }

    /// Maps `replacement`'s weights at the addresses of the ones they replace, on every rank's
    /// worker between two of its jobs, has the runners recompute what they derive from them,
    /// and returns each rank's report.
    fn rebind(&self, replacement: Replacement) -> Result<Vec<WeightsUpdated>> {
        let Replacement { target, ranks, .. } = replacement;

        // What a runner derives from its weights is computed in its graph pool, which a suspended
        // rank has unmapped: the restore recomputes it.
        let suspended = self.suspended.load(Ordering::Acquire);
        if suspended {
            self.derived_stale.store(true, Ordering::Release);
        }

        let moved = self.each("putting the new weights in place", |index, rank| {
            let weights = Arc::clone(&rank.weights);
            let pairs = ranks
                .get(index)
                .map(|share| share.pairs.clone())
                .unwrap_or_default();

            Box::new(move |runners: &mut [Box<dyn Runner>]| {
                let moved = weights.rebind(&pairs)?;
                // The graphs read the weights at the addresses that were mapped again.
                crate::profiling::storage_remapped();

                if !suspended {
                    for runner in runners.iter_mut() {
                        runner.weights_changed()?;
                    }
                }
                Ok(moved)
            })
        });
        let moved = match moved {
            Ok(moved) => moved,
            Err(error) => {
                *lock(&self.poisoned) = Some(error.to_string());
                return Err(error);
            }
        };

        for (rank, share) in self.ranks.iter().zip(ranks) {
            lock(rank).serving = share.order;
        }

        // The cached pages were computed from the weights this replaces, so later pages are keyed
        // by the new ones at every cache level: a prompt no longer finds the old.
        let pages = page_identity(&self.served_here());
        let mut reports = Vec::with_capacity(moved.len());
        for (rank, moved) in self.ranks.iter().zip(moved) {
            let rank = lock(rank);
            let identities = rank
                .cache
                .as_ref()
                .map(|cache| lock(cache).page_identities(&pages))
                .unwrap_or_default();

            reports.push(WeightsUpdated {
                group: rank.client.scope().map(|scope| scope.group.clone()),
                rank: rank.client.scope().map(|scope| scope.rank),
                identity: Some(identity_of(&rank.serving)),
                identities,
                moved,
            });
        }

        *lock(&self.target) = target;
        Ok(reports)
    }

    /// Suspends the graph pools of every rank between two of its worker's jobs: what a sleep
    /// gives back beside the cache, and what a load the free storage does not hold takes. The
    /// pools' reservation in the quota drops with them. The caller has paused the schedulers,
    /// since nothing may replay a graph until the restore.
    ///
    /// # Errors
    ///
    /// When a device refuses; the pools suspended before stay so, and a restore brings them back.
    pub(crate) fn suspend_graphs(&self) -> Result<()> {
        let answers = self.ask_peers(None, || DataOperation::SuspendGraphs)?;
        let here = self.suspend_here();
        both(here, answered::<()>(answers)).map(drop)
    }

    /// [`Self::suspend_graphs`] on this process's ranks.
    fn suspend_here(&self) -> Result<()> {
        if self.suspended.swap(true, Ordering::AcqRel) {
            return Ok(());
        }

        self.each("suspending the graph pools", |_, rank| {
            let graphs = Arc::clone(&rank.graphs);
            Box::new(move |_: &mut [Box<dyn Runner>]| {
                lock(&graphs).suspend()?;
                crate::profiling::storage_remapped();
                Ok(())
            })
        })?;

        tracing::info!("the graph pools gave their storage back at unchanged addresses");
        Ok(())
    }

    /// Restores what [`Self::suspend_graphs`] gave back, at the addresses the graphs hold, and
    /// has the runners recompute what they derive from weights that changed meanwhile. Each
    /// graph's executable is instantiated again before its next replay.
    ///
    /// # Errors
    ///
    /// When a device or the quota refuses, naming the pool; the restore may be tried again, and
    /// the schedulers stay paused until it succeeded.
    pub(crate) fn restore_graphs(&self) -> Result<()> {
        let answers = self.ask_peers(None, || DataOperation::RestoreGraphs)?;
        let here = self.restore_here();
        both(here, answered::<()>(answers)).map(drop)
    }

    /// [`Self::restore_graphs`] on this process's ranks.
    fn restore_here(&self) -> Result<()> {
        if !self.suspended.load(Ordering::Acquire) {
            return Ok(());
        }

        let stale = self.derived_stale.load(Ordering::Acquire);
        self.each("restoring the graph pools", |_, rank| {
            let graphs = Arc::clone(&rank.graphs);
            Box::new(move |runners: &mut [Box<dyn Runner>]| {
                lock(&graphs).restore()?;
                crate::profiling::storage_remapped();

                if stale {
                    for runner in runners.iter_mut() {
                        runner.weights_changed()?;
                    }
                }
                Ok(())
            })
        })?;

        self.derived_stale.store(false, Ordering::Release);
        self.suspended.store(false, Ordering::Release);
        tracing::info!("the graph pools took their storage back at the addresses the graphs hold");
        Ok(())
    }

    /// The ranks' side of a release from segment `first` on (module documentation): every rank
    /// of the worker groups `groups`, which state the device region `region` (the buffer pool's
    /// number of a KV region or a state region), unmaps the segments of its memory of the region
    /// from `first` on and ends its retention of them. The head has withdrawn them, so no
    /// region holds a unit of them and no job addresses them. Segments given back already are
    /// not given back twice. Returns, for this process (`None`) and each peer that holds such
    /// ranks (by its first rank), the CUDA ordinals of the devices of its ranks that gave
    /// segments back; a rank that keeps no cache, whose planes are not split or that holds no
    /// device names none.
    ///
    /// # Errors
    ///
    /// When the Storage Server refuses, here or on a peer's host.
    pub(crate) fn release_segments(
        &self,
        groups: &[GroupId],
        region: u8,
        first: u32,
    ) -> Result<Released> {
        let answers = self.ask_peers(Some(groups), || DataOperation::ReleaseSegments {
            groups: groups.to_vec(),
            region,
            from: first,
        })?;
        let peers: Vec<u32> = answers.iter().map(|(rank, _)| *rank).collect();
        let here = self
            .release_of(Some(groups), Some(region), first)
            .map(|tables| {
                // A rank whose planes are split gave segments back.
                tables
                    .iter()
                    .zip(self.devices_of(groups))
                    .filter_map(|(table, device)| table.as_ref().and(device))
                    .collect::<Vec<u32>>()
            });
        let (here, theirs) = both(here, answered::<Vec<u32>>(answers))?;

        let mut released = vec![(None, here)];
        released.extend(peers.into_iter().map(Some).zip(theirs));
        Ok(released)
    }

    /// [`Self::release_segments`] of device region `region` on the ranks of `groups`, or on
    /// every rank for `None`, and of every split region of each rank's cache for `None` (a
    /// sleep): each such rank's segment table of the region (of its KV region for every region);
    /// none for a rank that keeps no cache or whose memory of the region is not split, which has
    /// nothing to release.
    fn release_of(
        &self,
        groups: Option<&[GroupId]>,
        region: Option<u8>,
        first: u32,
    ) -> Result<Vec<Option<CacheTable>>> {
        self.each_of(groups, "releasing the cache's segments", |_, rank| {
            let cache = rank.cache.as_ref().map(Arc::clone);
            let client = Arc::clone(&rank.client);

            Box::new(move |_: &mut [Box<dyn Runner>]| {
                let Some(cache) = cache else {
                    return Ok(None);
                };
                let mut cache = lock(&cache);
                let split = cache.split_regions();
                let Some(&reported) = split.first() else {
                    return Ok(None);
                };
                let released: Vec<u8> = match region {
                    Some(region) if split.contains(&region) => vec![region],
                    Some(_) => return Ok(None),
                    None => split,
                };

                for region in &released {
                    cache.release_segments(*region, first, &*client)?;
                }
                // The graphs address the planes, whose tail was unmapped.
                crate::profiling::storage_remapped();
                Ok(segment_table(&cache, region.unwrap_or(reported)))
            })
        })
    }

    /// The CUDA ordinal of the device of each of this process's ranks of `groups`, in the order
    /// [`Self::each_of`] runs them; none for a rank of no device.
    fn devices_of(&self, groups: &[GroupId]) -> Vec<Option<u32>> {
        self.ranks
            .iter()
            .map(|rank| lock(rank))
            .filter(|rank| rank.of(groups))
            .map(|rank| {
                let worker = rank.worker.upgrade()?;
                u32::try_from(worker.device()?.0).ok()
            })
            .collect()
    }

    /// The ranks' side of creating given-back segments (module documentation): every rank of the
    /// worker groups `groups`, which state the device region `region`, creates the first `limit`
    /// segments of its memory of the region it gave back, or all of them, at their addresses, and
    /// resets the KV planes' units. Returns each such rank's segment table of the region, from
    /// which the head grants the segments held now; none for a rank that keeps no cache or whose
    /// memory of the region is not split.
    ///
    /// # Errors
    ///
    /// When the Storage Server or the device refuses, here or on a peer's host, a rank's planes
    /// then being as they were; or ranks whose tables disagree, of which the head grants nothing.
    pub(crate) fn create_segments(
        &self,
        groups: &[GroupId],
        region: u8,
        limit: Option<u32>,
    ) -> Result<Vec<Option<CacheTable>>> {
        let answers = self.ask_peers(Some(groups), || DataOperation::CreateSegments {
            groups: groups.to_vec(),
            region,
            count: limit,
        })?;
        let here = self.create_of(Some(groups), region, limit);
        let tables = joined_tables(here, answers)?;
        agreed(&tables)?;
        Ok(tables)
    }

    /// [`Self::create_segments`] on the ranks of `groups`, or on every rank for `None`.
    fn create_of(
        &self,
        groups: Option<&[GroupId]>,
        region: u8,
        limit: Option<u32>,
    ) -> Result<Vec<Option<CacheTable>>> {
        self.each_of(groups, "creating the cache's segments", |_, rank| {
            let cache = rank.cache.as_ref().map(Arc::clone);
            let client = Arc::clone(&rank.client);
            let stream = rank.stream.clone();

            Box::new(move |_: &mut [Box<dyn Runner>]| {
                let Some(cache) = cache else {
                    return Ok(None);
                };
                let mut cache = lock(&cache);

                cache.wake(region, &*client, &stream, limit)?;
                crate::profiling::storage_remapped();
                Ok(segment_table(&cache, region))
            })
        })
    }

    /// The segment table of device region `region` of every rank of the worker groups `groups`,
    /// the peers' included, as the rank holds it now, in rank order; none for a rank that keeps no
    /// cache or whose memory of the region is not split. What the head grants after a wake, which
    /// creates every rank's segments at once.
    ///
    /// # Errors
    ///
    /// A peer's failure.
    pub(crate) fn segment_tables(
        &self,
        groups: &[GroupId],
        region: u8,
    ) -> Result<Vec<Option<CacheTable>>> {
        let answers = self.ask_peers(Some(groups), || DataOperation::SegmentTables {
            groups: groups.to_vec(),
            region,
        })?;
        joined_tables(Ok(self.segment_tables_here(groups, region)), answers)
    }

    /// [`Self::segment_tables`] of this process's ranks.
    pub(crate) fn segment_tables_here(
        &self,
        groups: &[GroupId],
        region: u8,
    ) -> Vec<Option<CacheTable>> {
        self.ranks
            .iter()
            .map(|rank| lock(rank))
            .filter(|rank| rank.of(groups))
            .map(|rank| {
                let cache = rank.cache.as_ref()?;
                segment_table(&lock(cache), region)
            })
            .collect()
    }

    /// How many of the segments of device region `region` the ranks of the worker groups
    /// `groups` gave back, counted from the first of them, every such rank, the peers' included,
    /// has room for in its GPU's quota and on its device: what a grow creates
    /// ([`Self::create_segments`]). Zero when a rank has none given back or no room for the
    /// first, and when no rank of the groups splits its memory of the region.
    ///
    /// # Errors
    ///
    /// When a rank's quota or device cannot be read, here or on a peer's host.
    pub(crate) fn fitting_segments(&self, groups: &[GroupId], region: u8) -> Result<u32> {
        let answers = self.ask_peers(Some(groups), || DataOperation::FittingSegments {
            groups: groups.to_vec(),
            region,
        })?;
        let here = self.fitting_here(groups, region);
        let (here, peers) = both(here, answered::<Option<u32>>(answers))?;
        Ok(peers.into_iter().chain([here]).flatten().min().unwrap_or(0))
    }

    /// [`Self::fitting_segments`] over this process's ranks of `groups`; none when none of them
    /// splits its planes.
    ///
    /// # Errors
    ///
    /// When a rank's quota or device cannot be read.
    pub(crate) fn fitting_here(&self, groups: &[GroupId], region: u8) -> Result<Option<u32>> {
        let mut fitting: Option<u32> = None;
        for (_, room, table, unit_bytes) in self.segment_facts(groups, region)? {
            // The bytes the next given-back segments take on this rank, the first of them first.
            let given_back = table.segments.iter().skip(table.held as usize);
            let mut bytes: u64 = 0;
            let mut fits: u32 = 0;
            for segment in given_back {
                bytes = bytes.saturating_add(u64::from(segment.units).saturating_mul(unit_bytes));
                if !room.holds(bytes, 0) {
                    break;
                }
                fits += 1;
            }
            fitting = Some(fitting.map_or(fits, |fitting| fitting.min(fits)));
        }
        Ok(fitting)
    }

    /// How many of the segments of device region `region` the ranks of the worker groups `groups`
    /// hold, counted from the first, every such rank, the peers' included, holds beside its warm
    /// process in its GPU's quota: what a start keeps, once its ranks are warm, of the segments
    /// it created before its warm-up (`crate::process::epoch::keep_warm_segments`). None when no
    /// rank of the groups splits its memory of the region.
    ///
    /// # Errors
    ///
    /// When a rank's process, quota or segments cannot be read, here or on a peer's host.
    pub(crate) fn warm_segments(&self, groups: &[GroupId], region: u8) -> Result<Option<u32>> {
        let answers = self.ask_peers(Some(groups), || DataOperation::WarmSegments {
            groups: groups.to_vec(),
            region,
        })?;
        let here = self.warm_here(groups, region);
        let (here, peers) = both(here, answered::<Option<u32>>(answers))?;
        Ok(peers.into_iter().chain([here]).flatten().min())
    }

    /// [`Self::warm_segments`] over this process's ranks of `groups`. Each rank measures its
    /// process as [`Self::reserve_serving`] does, once its worker has passed every job before it,
    /// and keeps the segments it holds as far as its quota holds them beside that process
    /// ([`kept_beside`]). None when none of them splits its planes.
    ///
    /// # Errors
    ///
    /// When NVML cannot state a rank's process, or its quota cannot be read.
    pub(crate) fn warm_here(&self, groups: &[GroupId], region: u8) -> Result<Option<u32>> {
        // A process that holds no rank of the groups measures nothing.
        if !self.ranks.iter().any(|rank| lock(rank).of(groups)) {
            return Ok(None);
        }
        let nvml = Arc::new(Nvml::init()?);
        let devices = self.device_pools();
        let kept = self.each_of(
            Some(groups),
            "measuring the warm process beside the KV planes",
            |_, rank| {
                let nvml = Arc::clone(&nvml);
                let cache = rank.cache.as_ref().map(Arc::clone);
                let client = Arc::clone(&rank.client);
                let device = rank.device;
                let (pools, unattributed) = devices.get(&device).cloned().unwrap_or_default();

                Box::new(move |_: &mut [Box<dyn Runner>]| {
                    let Some(cache) = cache else {
                        return Ok(None);
                    };
                    let Some((table, unit_bytes)) = lock(&cache).segment_table(region) else {
                        return Ok(None);
                    };

                    // The process as its reservation will state it, and by how much the quota's
                    // reservations would pass its limit with it.
                    let process = measure(&nvml, device, &pools, unattributed)?;
                    let (quota, _) = client.usage(device)?;
                    let epoch = client.epoch();
                    let excess = division::excess(&quota, epoch, process)?;

                    let kept = kept_beside(&table, unit_bytes, epoch, excess);
                    if kept < table.held {
                        tracing::info!(
                            device = device.0,
                            process,
                            excess,
                            held = table.held,
                            kept,
                            "the warm process leaves the GPU's quota fewer of the cache's segments"
                        );
                    }
                    Ok(Some(kept))
                })
            },
        )?;
        Ok(kept.into_iter().flatten().min())
    }

    /// The room of the device of every rank of the worker groups `groups` whose memory of device
    /// region `region` is split, the peers' after this process's: its GPU's identity, its room,
    /// and the device bytes one unit of that memory takes, which a release of a unit gives back to
    /// the quota and the device alike. What a release to make room for other storage decides from; a GPU's
    /// identity names it on whichever host it is.
    ///
    /// # Errors
    ///
    /// When a rank's quota, device or identity cannot be read, here or on a peer's host.
    pub(crate) fn device_rooms(
        &self,
        groups: &[GroupId],
        region: u8,
    ) -> Result<Vec<([u8; 16], Room, u64)>> {
        let answers = self.ask_peers(Some(groups), || DataOperation::DeviceRooms {
            groups: groups.to_vec(),
            region,
        })?;
        let here = self.device_rooms_here(groups, region);
        let (mut rooms, peers) = both(here, answered::<Vec<([u8; 16], Room, u64)>>(answers))?;
        rooms.extend(peers.into_iter().flatten());
        Ok(rooms)
    }

    /// [`Self::device_rooms`] of this process's ranks.
    ///
    /// # Errors
    ///
    /// When a rank's quota, device or identity cannot be read.
    pub(crate) fn device_rooms_here(
        &self,
        groups: &[GroupId],
        region: u8,
    ) -> Result<Vec<([u8; 16], Room, u64)>> {
        Ok(self
            .segment_facts(groups, region)?
            .into_iter()
            .map(|(gpu, room, _, unit_bytes)| (gpu, room, unit_bytes))
            .collect())
    }

    /// What a segment decision reads of each rank of the worker groups `groups` whose memory of
    /// device region `region` is split, in rank order.
    fn segment_facts(&self, groups: &[GroupId], region: u8) -> Result<Vec<SegmentFacts>> {
        let mut facts = Vec::new();
        for rank in &self.ranks {
            let rank = lock(rank);
            if !rank.of(groups) {
                continue;
            }
            let Some(cache) = rank.cache.as_ref() else {
                continue;
            };
            let Some((table, unit_bytes)) = lock(cache).segment_table(region) else {
                continue;
            };

            facts.push((rank.device.uuid()?.0, room_of(&rank)?, table, unit_bytes));
        }
        Ok(facts)
    }

    /// The ranks' side of a sleep (module documentation): the graph pools are suspended and every
    /// segment of every rank's KV planes and state units is released. Returns each rank's segment
    /// table of its KV region.
    ///
    /// # Errors
    ///
    /// As [`Self::suspend_graphs`] and [`Self::release_segments`]; a wake undoes what was done.
    pub(crate) fn sleep(&self) -> Result<Vec<Option<CacheTable>>> {
        let answers = self.ask_peers(None, || DataOperation::Sleep)?;
        let here = self
            .suspend_here()
            .and_then(|()| self.release_of(None, None, 0));
        let tables = joined_tables(here, answers)?;

        tracing::info!("the ranks sleep: their graph storage and KV planes are given back");
        Ok(tables)
    }

    /// Has each rank's cache and state launch their own work from the generation the rank's
    /// graphs serve, from the rank's next job on: what the engine does after it switched or
    /// rolled back a generation, since no worker job sets it. Each rank takes its own device's
    /// generation, whose kernels its context loaded.
    ///
    /// # Errors
    ///
    /// When a rank's worker is gone, or a peer's failure.
    pub(crate) fn set_kernels(&self) -> Result<()> {
        let answers = self.ask_peers(None, || DataOperation::SetKernels)?;
        let here = self.set_kernels_here();
        both(here, answered::<()>(answers)).map(drop)
    }

    /// [`Self::set_kernels`] on this process's ranks.
    fn set_kernels_here(&self) -> Result<()> {
        self.each("setting the cache's and the state's kernels", |_, rank| {
            let cache = rank.cache.as_ref().map(Arc::clone);
            let state = rank.state.as_ref().map(Arc::clone);

            Box::new(move |runners: &mut [Box<dyn Runner>]| {
                // Every runner of a device serves one generation: the switch moves them together.
                let Some(kernels) = runners.first().map(|runner| runner.graphs().kernels()) else {
                    return Ok(());
                };
                if let Some(cache) = cache {
                    lock(&cache).set_kernels(Arc::clone(&kernels));
                }
                if let Some(state) = state {
                    lock(&state).set_kernels(kernels);
                }
                Ok(())
            })
        })?;
        Ok(())
    }

    /// Pairs a redistribution of the routed experts: checks `orders`, states what each rank needs,
    /// and holds the replacement until its copies start.
    ///
    /// # Errors
    ///
    /// When an order is no permutation of its layer's experts, or names no routed layer; nothing
    /// changed.
    pub(crate) fn prepare_experts(
        &self,
        orders: BTreeMap<String, Vec<u32>>,
    ) -> Result<Vec<String>> {
        let mut experts = lock(&self.experts);
        if experts.is_some() {
            return Err(Error::State("an expert redistribution is already prepared"));
        }

        let layers = {
            let rank = self
                .ranks
                .first()
                .ok_or(Error::State("no rank is served"))?;
            lock(rank).model.expert_layers()
        };
        for (layer, order) in &orders {
            let Some(count) = layers.get(layer) else {
                return Err(Error::Invalid(format!(
                    "{layer:?} is not a routed expert layer"
                )));
            };
            let distinct: BTreeSet<u32> = order.iter().copied().collect();
            if order.len() != *count as usize
                || distinct.len() != order.len()
                || distinct.iter().any(|expert| expert >= count)
            {
                return Err(Error::Invalid(format!(
                    "the order of {layer} is not a permutation of its {count} experts"
                )));
            }
        }

        let mut target = self.target();
        target.experts.extend(orders.clone());
        let replacement = self.pair_here(target, true)?;

        let mut reports = Vec::with_capacity(self.ranks.len());
        for (rank, share) in self.ranks.iter().zip(&replacement.ranks) {
            let device = lock(rank).device;
            reports.push(
                serde_json::json!({
                    "orders": orders,
                    "gpu": device.uuid()?.0,
                    "bytes": share.bytes,
                })
                .to_string(),
            );
        }

        *experts = Some(Experts::Paired(replacement));
        Ok(reports)
    }

    /// Starts the copies of the prepared redistribution, beside the serving batches.
    ///
    /// # Errors
    ///
    /// When no redistribution is prepared, or it started already.
    pub(crate) fn start_experts(self: &Arc<Self>) -> Result<Vec<String>> {
        let mut experts = lock(&self.experts);
        let replacement = match experts.take() {
            Some(Experts::Paired(replacement)) => replacement,
            other => {
                *experts = other;
                return Err(Error::State("no expert redistribution waits to start"));
            }
        };

        let plane = Arc::clone(self);
        let copying = std::thread::Builder::new()
            .name("expert-copies".into())
            .spawn(move || plane.load_here(replacement))
            .map_err(|error| rejected("starting the expert copies", error))?;

        *experts = Some(Experts::Copying(copying));
        Ok(self.ranks.iter().map(|_| "null".to_owned()).collect())
    }

    /// Whether the copies completed, as each rank responds.
    ///
    /// # Errors
    ///
    /// When a rank's copies failed; the redistribution is over, and the routing did not change.
    pub(crate) fn experts_ready(&self) -> Result<Vec<String>> {
        let mut experts = lock(&self.experts);
        let ready = match experts.take() {
            Some(Experts::Copying(task)) if task.is_finished() => {
                let replacement = task
                    .join()
                    .map_err(|_| Error::Invariant("the expert copies panicked".into()))??;
                *experts = Some(Experts::Ready(replacement));
                true
            }
            Some(Experts::Copying(task)) => {
                *experts = Some(Experts::Copying(task));
                false
            }
            Some(Experts::Ready(replacement)) => {
                *experts = Some(Experts::Ready(replacement));
                true
            }
            other => {
                *experts = other;
                return Err(Error::State("no expert copies run"));
            }
        };

        Ok(self.ranks.iter().map(|_| ready.to_string()).collect())
    }

    /// Ends a redistribution that will not commit: the incoming storage goes, and the routing and
    /// the weights that serve are untouched.
    ///
    /// # Errors
    ///
    /// When the copies panicked.
    pub(crate) fn abort_experts(&self) -> Result<Vec<String>> {
        let state = lock(&self.experts).take();
        match state {
            Some(Experts::Paired(replacement) | Experts::Ready(replacement)) => {
                self.discard_here(replacement);
            }
            Some(Experts::Copying(task)) => {
                // The copies own their allocations until they end, so they are waited for.
                if let Ok(replacement) = task
                    .join()
                    .map_err(|_| Error::Invariant("the expert copies panicked".into()))?
                {
                    self.discard_here(replacement);
                }
            }
            None => {}
        }

        Ok(self.ranks.iter().map(|_| "null".to_owned()).collect())
    }

    /// Puts the finished redistribution in place on every rank, the peers' included, each of
    /// which made its own, and returns each rank's report, in rank order. The caller has paused
    /// every scheduler, so the new order serves from one common boundary.
    ///
    /// # Errors
    ///
    /// As [`Self::commit_experts_here`], on any rank.
    pub(crate) fn commit_experts(
        &self,
        order: &BTreeMap<String, Vec<u32>>,
    ) -> Result<Vec<WeightsUpdated>> {
        let answers = self.ask_peers(None, || DataOperation::CommitExperts {
            order: order.clone(),
        })?;
        let here = self.commit_experts_here(order);
        self.reported(here, answers)
    }

    /// Puts the finished redistribution of this process's ranks in place and returns each rank's
    /// report.
    ///
    /// # Errors
    ///
    /// When its copies have not completed, or `order` is not the one they were made for; when a
    /// rank's mappings change part of the way, which poisons the epoch.
    pub(crate) fn commit_experts_here(
        &self,
        order: &BTreeMap<String, Vec<u32>>,
    ) -> Result<Vec<WeightsUpdated>> {
        let state = lock(&self.experts).take();
        let replacement = match state {
            Some(Experts::Ready(replacement)) => replacement,
            other => {
                *lock(&self.experts) = other;
                return Err(Error::State("the expert copies have not completed"));
            }
        };

        if replacement.target.experts != *order {
            self.discard_here(replacement);
            return Err(Error::Interface(
                "the committed order is not the one the copies were made for".into(),
            ));
        }
        self.rebind(replacement)
    }

    /// Publishes the LoRA adapter `payload` names into its slot, on every rank.
    ///
    /// The adapter's tensors take the path every weight takes: the Storage Server retains them
    /// under a key of their file and the rank maps them read-only, so a second engine of the
    /// deployment loading the same file shares them, and an unload returns them.
    ///
    /// # Errors
    ///
    /// When the payload names no path or slot, or the path holds no adapter; nothing stays loaded.
    pub(crate) fn load_adapter(&self, payload: &str) -> Result<Vec<String>> {
        let body: serde_json::Value = serde_json::from_str(payload)
            .map_err(|error| rejected("an adapter's payload", error))?;
        let slot = body["adapter"]
            .as_u64()
            .and_then(|slot| u32::try_from(slot).ok())
            .ok_or_else(|| Error::Invalid("load_lora_adapter names no slot".into()))?;
        let path = body["lora_path"]
            .as_str()
            .filter(|path| !path.is_empty())
            .ok_or_else(|| Error::Invalid("load_lora_adapter names no lora_path".into()))?;

        let checkpoint = nn::Checkpoint::open(std::path::Path::new(path))
            .map_err(|error| rejected("reading the adapter", error))?;
        if checkpoint.is_empty() {
            return Err(Error::Invalid(format!("{path} holds no adapter tensors")));
        }
        let names: Vec<String> = checkpoint.names().map(str::to_owned).collect();

        let mut adapters = Vec::with_capacity(self.ranks.len());
        let mut responses = Vec::with_capacity(self.ranks.len());
        for rank in &self.ranks {
            let (client, device) = {
                let rank = lock(rank);
                (Arc::clone(&rank.client), rank.device)
            };

            let loaded = WeightStore::new(client, device)
                .and_then(|store| Self::load_adapter_on(&checkpoint, &names, store));
            match loaded {
                Ok(adapter) => {
                    responses.push(
                        serde_json::json!({ "adapter": slot, "tensors": adapter.tensors.len() })
                            .to_string(),
                    );
                    adapters.push(adapter);
                }
                Err(error) => {
                    Self::unload(slot, adapters);
                    return Err(error);
                }
            }
        }

        tracing::info!(
            slot,
            path,
            tensors = names.len(),
            "loaded an adapter's tensors"
        );
        if let Some(replaced) = lock(&self.adapters).insert(slot, adapters) {
            Self::unload(slot, replaced);
        }
        Ok(responses)
    }

    /// Loads the tensors `names` of the adapter `checkpoint` through `store`, one rank's. A load
    /// that fails releases what it published.
    fn load_adapter_on(
        checkpoint: &nn::Checkpoint,
        names: &[String],
        store: WeightStore,
    ) -> Result<Adapter> {
        let mut tensors = BTreeMap::new();
        let loaded = (|| -> Result<()> {
            let loader = nn::WeightLoader::new(checkpoint, &store, None);
            for name in names {
                let shape = checkpoint
                    .tensor(name)
                    .ok_or_else(|| Error::Invariant(format!("no tensor {name}")))?
                    .shape
                    .clone();
                let source = nn::WeightSource::Region {
                    tensor: name.clone(),
                    start: 0,
                    shape: shape.clone(),
                };

                let tensor = loader.load(
                    name,
                    &source,
                    nn::WeightLayout::ROW_MAJOR,
                    &shape,
                    None,
                    false,
                )?;
                tensors.insert(name.clone(), tensor);
            }
            Ok(())
        })();

        // A load that failed publishes none of the allocations it packed: its weights are
        // forgotten.
        let finished = loaded.and_then(|()| store.finish());
        match finished {
            Ok(order) => Ok(Adapter {
                keys: order.into_iter().map(|(key, _)| key).collect(),
                tensors,
                store,
            }),
            Err(error) => {
                drop(tensors);
                let keys: Vec<WeightKey> =
                    store.abandon().into_iter().map(|(key, _)| key).collect();
                if let Err(cleanup) = store.forget(&keys) {
                    tracing::warn!(%cleanup, "a partial adapter load's storage was not released");
                }
                Err(error)
            }
        }
    }

    /// Releases every rank's tensors of the adapter in `slot`.
    fn unload(slot: u32, adapters: Vec<Adapter>) {
        for adapter in adapters {
            let Adapter {
                store,
                keys,
                tensors,
            } = adapter;

            // Nothing reads the tensors once they are dropped, and the mappings they lie over go
            // with the store.
            drop(tensors);
            if let Err(error) = store.forget(&keys) {
                tracing::warn!(%error, slot, "an adapter's storage was not released");
            }
        }
    }

    /// Releases the adapter of the slot `payload` names, on every rank.
    ///
    /// # Errors
    ///
    /// When the payload names no slot.
    pub(crate) fn unload_adapter(&self, payload: &str) -> Result<Vec<String>> {
        let body: serde_json::Value = serde_json::from_str(payload)
            .map_err(|error| rejected("an adapter's payload", error))?;
        let slot = body["adapter"]
            .as_u64()
            .and_then(|slot| u32::try_from(slot).ok())
            .ok_or_else(|| Error::Invalid("unload_lora_adapter names no slot".into()))?;

        // A slot with no adapter is a load that failed part of the way, which the daemon cleans
        // up after: nothing is held, and nothing is owed.
        let adapters = lock(&self.adapters).remove(&slot);
        if let Some(adapters) = adapters {
            Self::unload(slot, adapters);
        }

        Ok(self
            .ranks
            .iter()
            .map(|_| serde_json::json!({ "adapter": slot }).to_string())
            .collect())
    }
}

/// What a rank responds of a weight update: the JSON text of each is the update's `Hooked` result.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct WeightsUpdated {
    /// The rank's worker group and rank; `None` from a worker of no
    /// storage scope.
    #[serde(default)]
    pub(crate) group: Option<String>,
    #[serde(default)]
    pub(crate) rank: Option<u32>,
    /// The identity of the rank's new weights, which keys the trajectories
    /// they compute.
    #[serde(default)]
    pub(crate) identity: Option<String>,
    /// The page identity of each of the rank's cache groups under the new
    /// weights (`PrefixCacheInfo::identities`); empty for a rank that keeps
    /// no paged cache.
    #[serde(default)]
    pub(crate) identities: Vec<String>,
    /// The weights the update rewrote.
    #[serde(default)]
    pub(crate) moved: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::peers::link::Listener;
    use crate::executor::peers::messages::{MainProcessRequest, MainProcessResponse};
    use crate::executor::peers::{Liveness, accept, register};
    use crate::memory::cache::info::CacheSegment;
    use crate::process::generation;
    use std::thread::JoinHandle;
    use std::time::Duration;

    /// A bound on the tests' waits that only a defect reaches.
    const LONG: Duration = Duration::from_secs(30);

    /// A peer's link, with the groups it holds ranks of, and the thread that plays the peer.
    type Peer = ((Vec<GroupId>, Arc<Remote>), JoinHandle<Vec<DataOperation>>);

    /// The link to a peer engine process whose ranks are of the groups `held` and whose binding
    /// answers each operation it is sent with the JSON `answer` gives; the peer's thread returns
    /// the operations in the order they arrived, once the link closes.
    fn peer(
        rank: u32,
        held: Vec<GroupId>,
        answer: impl Fn(&DataOperation) -> String + Send + 'static,
    ) -> Peer {
        let listener = Listener::bind(&["127.0.0.1:0"]).unwrap();
        let address = listener.addresses()[0].clone();
        let thread = std::thread::spawn(move || {
            let mut channel = register(&address, rank, 1, LONG).unwrap();
            let mut operated = Vec::new();
            loop {
                let Some(request) = channel.recv::<MainProcessRequest>(LONG).unwrap() else {
                    continue;
                };
                match request {
                    MainProcessRequest::Data {
                        message_id,
                        operation,
                    } => {
                        let reply = answer(&operation);
                        operated.push(operation);
                        let answered = MainProcessResponse::Operated { message_id, reply };
                        channel.deliver(&answered).unwrap();
                    }
                    MainProcessRequest::Close { .. } => return operated,
                    other => panic!("the scheduler node sent {other:?}"),
                }
            }
        });

        let (accepted, channel) = accept(&listener, 1, 1, LONG).unwrap().pop().unwrap();
        let liveness = Liveness::new(Arc::new(|| {}), Duration::from_secs(3600));
        let remote = Remote::spawn(accepted, channel, liveness).unwrap();
        ((held, remote), thread)
    }

    /// The scheduler node's binding, whose own ranks hold no device (a meta start's), over the
    /// links of `peers`; and the operations each peer ran, once the links closed.
    fn over(peers: Vec<Peer>, check: impl FnOnce(&WeightBinding)) -> Vec<Vec<DataOperation>> {
        let (remotes, threads): (Vec<_>, Vec<_>) = peers.into_iter().unzip();
        let links: Vec<Arc<Remote>> = remotes
            .iter()
            .map(|(_, remote)| Arc::clone(remote))
            .collect();
        let binding = WeightBinding::new(Vec::new(), Target::default(), remotes);

        check(&binding);
        drop(binding);
        for link in links {
            link.close();
        }
        threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect()
    }

    /// The segment table of a rank whose planes are made of three segments of 16 units, of which
    /// it holds the first `held`.
    fn table(held: u32) -> CacheTable {
        let segment = |ordinal| CacheSegment {
            epoch: 1,
            ordinal,
            units: 16,
        };
        CacheTable {
            segments: (0..3).map(segment).collect(),
            held,
        }
    }

    /// The JSON of `value`, as a peer answers it.
    fn json(value: &impl Serialize) -> String {
        serde_json::to_string(value).unwrap()
    }

    /// The room a grow creates segments in leaves the Transfer Server's context, which the server
    /// reserves again before its next task on the GPU, whether it holds it now or not: segments of
    /// one granule a column fill the quota to within one of them, so a room that counted an idle
    /// context as free would have the quota refuse the context afterwards.
    #[test]
    fn a_device_s_room_leaves_the_transfer_server_s_context_held_or_not() {
        const GIB: u64 = 1 << 30;
        let reservation = |item, bytes| storage::protocol::Reservation {
            item,
            scope: None,
            epochs: Vec::new(),
            segment: None,
            bytes,
        };
        let quota = |reservations| storage::Quota {
            gpu: [0; 16],
            deployment: "model".into(),
            total: 2 * 70 * GIB,
            limit: Some(70 * GIB),
            reservations,
            host_link_bytes_per_second: None,
            transfer_context: GIB,
        };

        // 60 GiB of weights: the 1 GiB context leaves 9 GiB of the quota's other 10 whether it is
        // idle or held, when it is among the reservations and counted once.
        let idle = quota(vec![reservation(Item::Weights, 60 * GIB)]);
        assert_eq!(Room::of(&idle, 100 * GIB, 1).quota, 9 * GIB);
        let held = quota(vec![
            reservation(Item::Weights, 60 * GIB),
            reservation(Item::TransferContext, GIB),
        ]);
        assert_eq!(Room::of(&held, 100 * GIB, 1).quota, 9 * GIB);

        // The device's free bytes bound a load too, and a quota of no limit leaves those alone.
        assert!(!Room::of(&idle, 8 * GIB, 1).holds(9 * GIB, 0));
        let unlimited = storage::Quota {
            limit: None,
            ..idle
        };
        assert_eq!(Room::of(&unlimited, 8 * GIB, 1).quota, 8 * GIB);
    }

    /// A warm rank keeps the segments its quota holds beside its warm process: it gives back the
    /// fewest of its own at the tail whose bytes cover the excess, and never a segment another
    /// epoch created, whose storage stays that epoch's and frees nothing of the quota.
    #[test]
    fn a_warm_rank_gives_back_the_fewest_tail_segments_that_cover_its_excess() {
        const UNIT: u64 = 1 << 20;
        let segment = 16 * UNIT;

        // Nothing past the limit: every held segment stays.
        assert_eq!(kept_beside(&table(3), UNIT, 1, 0), 3);
        // A byte past the limit takes a whole segment back, and a segment's bytes exactly one.
        assert_eq!(kept_beside(&table(3), UNIT, 1, 1), 2);
        assert_eq!(kept_beside(&table(3), UNIT, 1, segment), 2);
        assert_eq!(kept_beside(&table(3), UNIT, 1, segment + 1), 1);
        // A segment given back already is not given back again.
        assert_eq!(kept_beside(&table(2), UNIT, 1, segment + 1), 0);
        // Segments another epoch created stay, whatever the excess.
        assert_eq!(kept_beside(&table(3), UNIT, 2, 3 * segment), 3);
    }

    /// The ranks of a region keep, once warm, the fewest segments any of them, a peer's included,
    /// holds beside its warm process; a peer whose ranks split no plane states none.
    #[test]
    fn a_warm_region_keeps_the_fewest_segments_any_rank_holds_beside_its_process() {
        const PREFILL: GroupId = GroupId(0);
        const DECODE: GroupId = GroupId(1);
        let kept = |kept: Option<u32>| {
            move |operation: &DataOperation| match operation {
                DataOperation::WarmSegments { .. } => json(&kept),
                other => panic!("the peer was sent {other:?}"),
            }
        };

        let operated = over(
            vec![
                peer(1, vec![PREFILL], kept(None)),
                peer(2, vec![PREFILL, DECODE], kept(Some(40))),
                peer(3, vec![DECODE], kept(Some(38))),
            ],
            |binding| {
                assert_eq!(binding.warm_segments(&[PREFILL], 0).unwrap(), Some(40));
                assert_eq!(binding.warm_segments(&[DECODE], 0).unwrap(), Some(38));
            },
        );

        // Each region asked only the peers that hold its ranks.
        let asked: Vec<usize> = operated.iter().map(Vec::len).collect();
        assert_eq!(asked, [1, 2, 1]);
    }

    #[test]
    fn the_segments_that_fit_are_the_fewest_any_rank_a_peer_s_included_has_room_for() {
        const PREFILL: GroupId = GroupId(0);
        const DECODE: GroupId = GroupId(1);
        let fitting = |fits: Option<u32>| {
            move |operation: &DataOperation| match operation {
                DataOperation::FittingSegments { .. } => json(&fits),
                other => panic!("the peer was sent {other:?}"),
            }
        };

        let operated = over(
            vec![
                // A peer of the prefill group whose ranks split no plane states none.
                peer(1, vec![PREFILL], fitting(None)),
                peer(2, vec![PREFILL, DECODE], fitting(Some(3))),
                // A peer of the decode group alone has no room, which binds no prefill rank.
                peer(3, vec![DECODE], fitting(Some(0))),
            ],
            |binding| {
                assert_eq!(binding.fitting_segments(&[PREFILL], 0).unwrap(), 3);
                assert_eq!(binding.fitting_segments(&[DECODE], 0).unwrap(), 0);
            },
        );

        // Each link was asked about the groups it holds ranks of only.
        let asked: Vec<usize> = operated.iter().map(Vec::len).collect();
        assert_eq!(asked, [1, 2, 1]);
    }

    #[test]
    fn segments_created_on_ranks_whose_tables_disagree_are_refused() {
        const GROUP: GroupId = GroupId(0);
        // Two peers create as many segments as asked; one of them holds one more than asked
        // once asked for one.
        let created = |behind: bool| {
            move |operation: &DataOperation| match operation {
                DataOperation::CreateSegments {
                    count: Some(count), ..
                } => {
                    let held = if behind && *count == 1 { 2 } else { *count };
                    json(&vec![Some(table(held))])
                }
                other => panic!("the peer was sent {other:?}"),
            }
        };

        over(
            vec![
                peer(1, vec![GROUP], created(false)),
                peer(2, vec![GROUP], created(true)),
            ],
            |binding| {
                // Tables that agree are every rank's, in rank order.
                let tables = binding.create_segments(&[GROUP], 0, Some(2)).unwrap();
                assert_eq!(tables, [Some(table(2)), Some(table(2))]);

                // The head grants a segment only once every rank that maps the region holds it.
                let refused = binding.create_segments(&[GROUP], 0, Some(1)).unwrap_err();
                assert!(
                    refused.to_string().contains("different segment tables"),
                    "{refused}"
                );
            },
        );
    }

    #[test]
    fn a_sleep_releases_every_peer_s_segments() {
        // A peer's ranks suspend their graph pools and give every segment back.
        let sleeping = |operation: &DataOperation| match operation {
            DataOperation::Sleep => json(&vec![Some(table(0))]),
            other => panic!("the peer was sent {other:?}"),
        };

        let operated = over(
            vec![
                peer(1, vec![GroupId(0)], sleeping),
                peer(2, vec![GroupId(1)], sleeping),
            ],
            |binding| {
                let tables = binding.sleep().unwrap();
                assert_eq!(tables, [Some(table(0)), Some(table(0))]);
            },
        );
        assert_eq!(operated, [[DataOperation::Sleep], [DataOperation::Sleep]]);
    }

    #[test]
    fn a_preparation_reports_what_peers_ranks_did_not_compare_once_the_shadow_pass_ended() {
        const GENERATION: u64 = 3;
        let entry = "operator `softmax` is new in generation 3".to_owned();
        // A peer's preparation answers once it ended; what its ranks could not compare, the shadow
        // batches that reached them included, it states when asked.
        let answer = {
            let entry = entry.clone();
            move |operation: &DataOperation| match operation {
                DataOperation::Prepared {
                    generation: GENERATION,
                } => json(&()),
                DataOperation::Uncompared => json(&vec![entry.clone()]),
                other => panic!("the peer was sent {other:?}"),
            }
        };

        let operated = over(
            vec![
                peer(1, vec![GroupId(0)], answer.clone()),
                peer(2, vec![GroupId(1)], answer),
            ],
            |binding| {
                // An update that replaced an operator: every peer's ranks are asked once the pass
                // ended, and each entry is reported once.
                let changed = ["softmax".to_owned()];
                let uncompared =
                    generation::compare(&[], &[], binding, GENERATION, &changed, LONG).unwrap();
                assert_eq!(uncompared, [entry.as_str()]);

                // An update that replaced none compares nothing.
                let none = generation::compare(&[], &[], binding, GENERATION, &[], LONG).unwrap();
                assert!(none.is_empty());
            },
        );

        // Each peer held the generation before any shadow batch, and was asked what its ranks did
        // not compare only after the pass.
        let prepared = DataOperation::Prepared {
            generation: GENERATION,
        };
        let asked = vec![prepared.clone(), DataOperation::Uncompared, prepared];
        assert_eq!(operated, [asked.clone(), asked]);
    }

    #[test]
    fn a_load_fits_both_the_quota_and_the_device() {
        let room = Room {
            quota: 100,
            free: 60,
            elsewhere: 0,
        };
        assert!(room.holds(60, 0));
        assert!(!room.holds(61, 0));
        // The graph storage gives back device storage and quota alike.
        assert!(room.holds(100, 40));
        assert!(!room.holds(101, 40));
    }

    #[test]
    fn one_set_of_weights_has_one_identity_whatever_loads_it() {
        let weight = |tensor: &str| {
            let key = WeightKey {
                files: Vec::new(),
                dtype: nn::DType::BF16,
                layout: nn::WeightLayout::ROW_MAJOR,
                tensors: vec![tensor.to_owned()],
                start: 0,
                shape: nn::Shape::from(vec![1]),
                stride: None,
                provenance: None,
            };
            (key, false)
        };

        let first = identity_of(&[weight("a"), weight("b")]);
        assert_eq!(first, identity_of(&[weight("a"), weight("b")]));
        assert_ne!(first, identity_of(&[weight("a"), weight("c")]));
    }
}

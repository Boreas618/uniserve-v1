//! The data plane of a serving epoch: the changes the engine makes to what its ranks hold on
//! their devices, inside the epoch.
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
//! - **At a batch boundary**: putting loaded weights in place ([`DataPlane::update_weights`],
//!   [`DataPlane::commit_experts`]), suspending and restoring the graph pools, releasing and
//!   creating the KV planes' segments, and the kernels the cache and the state launch, each
//!   rank's from the generation its graphs serve ([`DataPlane::set_kernels`]). A control
//!   operation is ordered against its own worker's jobs only, so the engine first pauses the
//!   schedulers whose batches read what changes: every rank of a group then changes between the
//!   same two batches.
//! - **Beside the serving batches**: a load of another checkpoint of the same model, or of
//!   another arrangement of its experts ([`DataPlane::pair`], [`DataPlane::load`]), published at
//!   the Storage Server while requests go on; LoRA adapters; profiling sessions, which each
//!   rank's worker starts and ends between two of its batches.
//! - **The quota**: what the process holds on each device beside its pools and the Storage
//!   Server's storages, reserved in the GPU's quota ([`DataPlane::reserve_serving`]) with what a
//!   profiling session's collectors hold.
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
//! those of the worker groups that state it, which the segment operations name. The data plane is
//! the ranks' side of each sequence:
//!
//! 1. **Release from segment `k` on**, for a region of another layout to grow into the memory:
//!    the schedulers `Resize` their regions below the first unit of segment `k`; the head sends
//!    `Withdraw` for each segment from `k` on, which takes its units out of the space's free
//!    ranges; then [`DataPlane::release_segments`]: every rank of the region unmaps them and ends
//!    its retention, and the GPU's quota usage drops with the last rank.
//! 2. **Create given-back segments**, for a region to grow past the space's free ranges, as many
//!    as [`DataPlane::fitting_segments`] finds room for: [`DataPlane::create_segments`]: every
//!    rank of the region creates them again at their addresses and resets their units; the head
//!    sends `Grant` for each segment the ranks hold now beyond those the space held; then the
//!    schedulers `Resize` their regions into them.
//! 3. **Sleep**: the schedulers detach every owner and `Resize` every device region to nothing,
//!    having stored what the write policy keeps; the head withdraws every segment;
//!    [`DataPlane::sleep`] releases them all and suspends the graph pools.
//! 4. **Wake**: [`DataPlane::wake`] restores the graph pools and creates every segment again; the
//!    head grants them ([`DataPlane::segment_tables`] states each region's); the schedulers grow
//!    their regions.
//!
//! Each returns its ranks' segment tables ([`crate::memory::cache::info::CacheTable`]), whose
//! segment `i` holds the units from the sum of the units of the segments before it. An epoch switch
//! changes no plane: two epochs of one layout name the same regions and map the same segments, and
//! nothing of the cache is handed over or adopted (`specs/updates.md` §3.4).

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

use crate::memory::cache::KvCacheMemory;
use crate::memory::cache::info::CacheTable;
use crate::memory::client::{StorageClient, quota_limit};
use crate::memory::division::process_bytes;
use crate::memory::graph_pool::GraphPool;
use crate::memory::reload::{MetaWeights, Reload, paired};
use crate::memory::state::StateMemory;
use crate::memory::weights::WeightStore;
use crate::profiling::session::Change;
use crate::worker::completion::Outcome;
use nn::WeightKey;
use platform::{DeviceId, Nvml, Stream};
use proto::request::wire::{DeviceMemory, PoolMemory};
use serde::{Deserialize, Serialize};
use service::lock;
use storage::Item;
use storage::runtime::GroupId;

use crate::runner::Runner;
use crate::worker::{Error, Result, Worker};

/// The name [`DataPlane::memory`] reports the planes of a device's KV regions under, beside the
/// pools of its allocators (`graph`, `scratch`, `resident`): the segments the ranks map, with the
/// flags, scales, index keys and state planes beside them.
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

/// What the data plane holds of one rank: the owners of its device's storage, which the rank's
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
    /// A stream of the device that only the data plane's control operations issue on: the resets
    /// of the units of segments created again.
    pub stream: Stream,
    /// The device bytes of the rank's communicators that NVML attributes to no process, as their
    /// owner states them: what a measurement of the process adds to NVML's figure.
    pub unattributed: u64,
    /// How the rank loads its model.
    pub model: Arc<dyn ModelLoad>,
    /// The weights the rank serves, in its load's order, each with whether the model writes it
    /// after the load.
    pub serving: Vec<(WeightKey, bool)>,
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
    /// Whether the new weights are published yet, so a discard knows what there is to release.
    loaded: bool,
}

impl Replacement {
    /// The most device bytes any rank needs beside what it serves.
    pub(crate) fn bytes(&self) -> u64 {
        self.ranks.iter().map(|rank| rank.bytes).max().unwrap_or(0)
    }
}

/// The room one device has beside what it holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Room {
    /// Bytes the deployment's quota of the GPU leaves.
    pub quota: u64,
    /// Bytes the device has free.
    pub free: u64,
}

impl Room {
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

/// The data plane of one process: every rank it holds.
pub(crate) struct DataPlane {
    ranks: Vec<Mutex<Rank>>,
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
    let mut digest = ::storage::key::Digest::new(b"uniserve-weights-identity");
    for (key, _) in order {
        digest.text(&key.to_string());
    }
    ::storage::Key(digest.finish()).hex()
}

/// The room `rank`'s device has: what the deployment's quota of the GPU leaves, and what the
/// device has free.
fn room_of(rank: &Rank) -> Result<Room> {
    let (quota, _) = rank.client.usage(rank.device)?;
    let free = rank.device.memory_info()?.free;

    Ok(Room {
        quota: quota
            .limit
            .map_or(free, |limit| limit.saturating_sub(quota.usage())),
        free,
    })
}

/// What `cache`'s planes are made of and how many of their segments exist; none for planes that
/// are not split.
fn segment_table(cache: &KvCacheMemory) -> Option<CacheTable> {
    cache.segments().map(|segments| CacheTable {
        segments: segments.to_vec(),
        held: cache.held_segments(),
    })
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

impl DataPlane {
    /// The data plane over `ranks`, which serve `target`.
    pub(crate) fn new(ranks: Vec<Rank>, target: Target) -> Self {
        Self {
            ranks: ranks.into_iter().map(Mutex::new).collect(),
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
                        // A data plane that stopped waiting has nothing to be told.
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

    /// Reserves what each rank's process holds beside its pools and the Storage Server's storages
    /// in its device's quota, as the ranks start to serve: its CUDA context, its kernels and its
    /// communicators, measured once the worker has passed every job before it. From here on the
    /// quota binds the whole rank, so a storage that would take the deployment past it is
    /// refused at the Storage Server and not found out by the device.
    ///
    /// Reserved once: a process that already reserved keeps its figures, and a standby reserves
    /// when it is promoted, because until then the quota counts what it maps of its predecessor.
    ///
    /// # Errors
    ///
    /// When a rank's process does not fit its quota, or NVML cannot state the process's storage.
    pub(crate) fn reserve_serving(&self) -> Result<()> {
        let mut reserved = lock(&self.reserved);
        // A data plane of no rank has no process storage to measure.
        if !reserved.is_empty() || self.ranks.is_empty() {
            return Ok(());
        }

        let nvml = Arc::new(Nvml::init()?);
        let measured = self.each("measuring the process's device storage", |_, rank| {
            let nvml = Arc::clone(&nvml);
            let graphs = Arc::clone(&rank.graphs);
            let (device, unattributed) = (rank.device, rank.unattributed);

            Box::new(move |_: &mut [Box<dyn Runner>]| {
                let pools = lock(&graphs).mapped();
                process_bytes(&nvml, device, pools, unattributed)
            })
        })?;

        let reservations: Vec<Reservation> = measured
            .into_iter()
            .map(|process| Reservation {
                process,
                profiling: 0,
            })
            .collect();
        for (rank, reservation) in self.ranks.iter().zip(&reservations) {
            self.reserve(&lock(rank), reservation)?;
        }

        *reserved = reservations;
        Ok(())
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

    /// What the rank's pages are computed from, which keys them in the host KV cache: the identity
    /// of the weights it serves. An epoch of the same weights finds the pages another left there,
    /// and one of other weights, or this one after a weight update, keys its own.
    ///
    /// # Errors
    ///
    /// When this data plane holds no rank.
    #[expect(dead_code, reason = "the identity is read through each rank's report")]
    pub(crate) fn serving_identity(&self) -> Result<String> {
        if self.ranks.is_empty() {
            return Err(Error::State("the data plane holds no rank"));
        }
        Ok(self.identity_of_ranks())
    }

    /// The identity of every rank's weights, in rank order. What a rank's pages hold follows from
    /// all of them: an expert union's rank computes each token's experts on its peers, so a page is
    /// another page once any rank's weights change, and every rank keys it alike.
    fn identity_of_ranks(&self) -> String {
        let order: Vec<(WeightKey, bool)> = self
            .ranks
            .iter()
            .flat_map(|rank| lock(rank).serving.clone())
            .collect();
        identity_of(&order)
    }

    /// The room every device has for a load, in rank order.
    ///
    /// # Errors
    ///
    /// When the quota or the device cannot be read.
    pub(crate) fn room(&self) -> Result<Vec<Room>> {
        self.ranks.iter().map(|rank| room_of(&lock(rank))).collect()
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

    /// The bytes of device storage the graph pools map, on the device that maps the fewest: what
    /// a suspension of the pools gives back on every device.
    pub(crate) fn graph_bytes(&self) -> u64 {
        self.ranks
            .iter()
            .map(|rank| Arc::clone(&lock(rank).graphs))
            .map(|graphs| lock(&graphs).mapped())
            .min()
            .unwrap_or(0)
    }

    /// Pairs the weights of a load of `target` with the ones that serve, on every rank, moving no
    /// byte: the model's load runs over storage on the meta device, which states the weights it
    /// reaches without reading a payload. With `rearranged`, a stack may hold the same parts in
    /// another order (an expert redistribution); without, each weight reads the same tensors (a
    /// weight update).
    ///
    /// # Errors
    ///
    /// When the target cannot be the served model: the sentence names its first difference.
    pub(crate) fn pair(&self, target: Target, rearranged: bool) -> Result<Replacement> {
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
            loaded: false,
        })
    }

    /// Publishes what `replacement` changes on every rank, each rank on a thread of its own: the
    /// batches go on reading the weights that serve.
    ///
    /// # Errors
    ///
    /// When a rank's load fails: everything this load published is released, and the ranks serve
    /// on as they did.
    pub(crate) fn load(&self, mut replacement: Replacement) -> Result<Replacement> {
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
            self.discard(replacement);
            return Err(Error::Invalid(format!(
                "the weights of {} did not load: {failure}",
                target.checkpoint.display()
            )));
        }
        Ok(replacement)
    }

    /// Ends a replacement that will not serve: what its load published goes back, and the ranks
    /// serve on.
    pub(crate) fn discard(&self, replacement: Replacement) {
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
    /// ([`Self::update_weights`]).
    pub(crate) fn stage(&self, replacement: Replacement) {
        if let Some(previous) = lock(&self.staged).replace(replacement) {
            self.discard(previous);
        }
    }

    /// Ends a staged replacement that will not serve.
    #[expect(dead_code, reason = "no job abandons a staged replacement")]
    pub(crate) fn discard_staged(&self) {
        if let Some(replacement) = lock(&self.staged).take() {
            self.discard(replacement);
        }
    }

    /// Puts the staged replacement in place on every rank and returns each rank's report.
    ///
    /// The caller has paused every scheduler whose batches read the ranks' weights, so every
    /// rank of a group changes between the same two batches.
    ///
    /// # Errors
    ///
    /// When no replacement is staged, and as [`Self::commit_experts`].
    pub(crate) fn update_weights(&self) -> Result<Vec<WeightsUpdated>> {
        let replacement = lock(&self.staged).take().ok_or(Error::State(
            "no weights are loaded for this update to put in place",
        ))?;
        self.rebind(replacement)
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
        let pages = self.identity_of_ranks();
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
    /// has the runners recompute what they derive from weights that changed meanwhile. The
    /// workers renew every graph before its next replay.
    ///
    /// # Errors
    ///
    /// When a device or the quota refuses, naming the pool; the restore may be tried again, and
    /// the schedulers stay paused until it succeeded.
    pub(crate) fn restore_graphs(&self) -> Result<()> {
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
    /// of the worker groups `groups`, which state the region, unmaps the segments of its KV
    /// planes from `first` on and ends its retention of them. The head has withdrawn them, so no
    /// region holds a unit of them and no job addresses them. Segments given back already are
    /// not given back twice. Returns each such rank's segment table; none for a rank that keeps
    /// no cache or whose planes are not split, which has nothing to release.
    ///
    /// # Errors
    ///
    /// When the Storage Server refuses.
    pub(crate) fn release_segments(
        &self,
        groups: &[GroupId],
        first: u32,
    ) -> Result<Vec<Option<CacheTable>>> {
        self.release_of(Some(groups), first)
    }

    /// [`Self::release_segments`] on the ranks of `groups`, or on every rank for `None`.
    fn release_of(
        &self,
        groups: Option<&[GroupId]>,
        first: u32,
    ) -> Result<Vec<Option<CacheTable>>> {
        self.each_of(groups, "releasing the KV planes' segments", |_, rank| {
            let cache = rank.cache.as_ref().map(Arc::clone);
            let client = Arc::clone(&rank.client);

            Box::new(move |_: &mut [Box<dyn Runner>]| {
                let Some(cache) = cache else {
                    return Ok(None);
                };
                let mut cache = lock(&cache);
                if cache.segments().is_none() {
                    return Ok(None);
                }

                cache.release_segments(first, &*client)?;
                // The graphs address the planes, whose tail was unmapped.
                crate::profiling::storage_remapped();
                Ok(segment_table(&cache))
            })
        })
    }

    /// The ranks' side of creating given-back segments (module documentation): every rank of the
    /// worker groups `groups`, which state the region, creates the first `limit` segments of its
    /// KV planes it gave back, or all of them, at their addresses, and resets their units.
    /// Returns each such rank's segment table, from which the head grants the segments held now;
    /// none for a rank that keeps no cache or whose planes are not split.
    ///
    /// # Errors
    ///
    /// When the Storage Server or the device refuses; a rank's planes are then as they were.
    pub(crate) fn create_segments(
        &self,
        groups: &[GroupId],
        limit: Option<u32>,
    ) -> Result<Vec<Option<CacheTable>>> {
        self.create_of(Some(groups), limit)
    }

    /// [`Self::create_segments`] on the ranks of `groups`, or on every rank for `None`.
    fn create_of(
        &self,
        groups: Option<&[GroupId]>,
        limit: Option<u32>,
    ) -> Result<Vec<Option<CacheTable>>> {
        self.each_of(groups, "creating the KV planes' segments", |_, rank| {
            let cache = rank.cache.as_ref().map(Arc::clone);
            let client = Arc::clone(&rank.client);
            let stream = rank.stream.clone();

            Box::new(move |_: &mut [Box<dyn Runner>]| {
                let Some(cache) = cache else {
                    return Ok(None);
                };
                let mut cache = lock(&cache);

                cache.wake(&*client, &stream, limit)?;
                crate::profiling::storage_remapped();
                Ok(segment_table(&cache))
            })
        })
    }

    /// The segment table of the KV planes of every rank of the worker groups `groups`, as the
    /// rank holds it now, in rank order; none for a rank that keeps no cache or whose planes are
    /// not split. What the head grants after a wake, which creates every rank's segments at once.
    pub(crate) fn segment_tables(&self, groups: &[GroupId]) -> Vec<Option<CacheTable>> {
        self.ranks
            .iter()
            .map(|rank| lock(rank))
            .filter(|rank| rank.of(groups))
            .map(|rank| {
                let cache = rank.cache.as_ref()?;
                segment_table(&lock(cache))
            })
            .collect()
    }

    /// How many of the segments the ranks of the worker groups `groups` gave back, counted from
    /// the first of them, every such rank has room for in its GPU's quota and on its device: what
    /// a grow creates ([`Self::create_segments`]). Zero when a rank has none given back or no
    /// room for the first, and when no rank of the groups splits its planes.
    ///
    /// # Errors
    ///
    /// When a rank's quota or device cannot be read.
    pub(crate) fn fitting_segments(&self, groups: &[GroupId]) -> Result<u32> {
        let mut fitting: Option<u32> = None;
        for (_, room, table, unit_bytes) in self.segment_facts(groups)? {
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
        Ok(fitting.unwrap_or(0))
    }

    /// The room of the device of every rank of the worker groups `groups` whose KV planes are
    /// split, in rank order: its GPU's identity, its room, and the device bytes one unit of its
    /// planes takes, which a release of a unit gives back to the quota and the device alike. What
    /// a release to make room for other storage decides from.
    ///
    /// # Errors
    ///
    /// When a rank's quota, device or identity cannot be read.
    pub(crate) fn device_rooms(&self, groups: &[GroupId]) -> Result<Vec<([u8; 16], Room, u64)>> {
        Ok(self
            .segment_facts(groups)?
            .into_iter()
            .map(|(gpu, room, _, unit_bytes)| (gpu, room, unit_bytes))
            .collect())
    }

    /// What a segment decision reads of each rank of the worker groups `groups` whose KV planes
    /// are split, in rank order.
    fn segment_facts(&self, groups: &[GroupId]) -> Result<Vec<SegmentFacts>> {
        let mut facts = Vec::new();
        for rank in &self.ranks {
            let rank = lock(rank);
            if !rank.of(groups) {
                continue;
            }
            let Some(cache) = rank.cache.as_ref() else {
                continue;
            };
            let (table, unit_bytes) = {
                let cache = lock(cache);
                (segment_table(&cache), cache.unit_bytes())
            };
            let Some(table) = table else {
                continue;
            };

            facts.push((rank.device.uuid()?.0, room_of(&rank)?, table, unit_bytes));
        }
        Ok(facts)
    }

    /// The ranks' side of a sleep (module documentation): the graph pools are suspended and every
    /// segment of every rank's KV planes is released. Returns each rank's segment table.
    ///
    /// # Errors
    ///
    /// As [`Self::suspend_graphs`] and [`Self::release_segments`]; a wake undoes what was done.
    pub(crate) fn sleep(&self) -> Result<Vec<Option<CacheTable>>> {
        self.suspend_graphs()?;
        let tables = self.release_of(None, 0)?;

        tracing::info!("the ranks sleep: their graph storage and KV planes are given back");
        Ok(tables)
    }

    /// The ranks' side of a wake (module documentation): what [`Self::sleep`] gave back is mapped
    /// again on every rank. Returns each rank's segment table, from which the head grants every
    /// segment.
    ///
    /// # Errors
    ///
    /// As [`Self::restore_graphs`] and [`Self::create_segments`].
    pub(crate) fn wake(&self) -> Result<Vec<Option<CacheTable>>> {
        self.restore_graphs()?;
        self.create_of(None, None)
    }

    /// Has each rank's cache and state launch their own work from the generation the rank's
    /// graphs serve, from the rank's next job on: what the engine does after it switched or
    /// rolled back a generation, since no worker job sets it. Each rank takes its own device's
    /// generation, whose kernels its context loaded.
    ///
    /// # Errors
    ///
    /// When a rank's worker is gone.
    pub(crate) fn set_kernels(&self) -> Result<()> {
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
        let replacement = self.pair(target, true)?;

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
            .spawn(move || plane.load(replacement))
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
                self.discard(replacement);
            }
            Some(Experts::Copying(task)) => {
                // The copies own their allocations until they end, so they are waited for.
                if let Ok(replacement) = task
                    .join()
                    .map_err(|_| Error::Invariant("the expert copies panicked".into()))?
                {
                    self.discard(replacement);
                }
            }
            None => {}
        }

        Ok(self.ranks.iter().map(|_| "null".to_owned()).collect())
    }

    /// Puts the finished redistribution in place on every rank and returns each rank's report.
    /// The caller has paused every scheduler, so the new order serves from one common boundary.
    ///
    /// # Errors
    ///
    /// When its copies have not completed, or `order` is not the one they were made for; when a
    /// rank's mappings change part of the way, which poisons the epoch.
    pub(crate) fn commit_experts(
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
            self.discard(replacement);
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
            let loader = nn::WeightLoader::new(checkpoint, &store, None, None);
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

    #[test]
    fn a_load_fits_both_the_quota_and_the_device() {
        let room = Room {
            quota: 100,
            free: 60,
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

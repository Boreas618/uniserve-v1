//! The data plane of a serving epoch (`specs/worker.md` §7.6): the updates that change what a
//! rank holds at the Storage Server, inside the epoch and without going through the three update
//! layers.
//!
//! Two kinds of work live here, and the difference is who may run them when:
//!
//! - **Storage operations at a batch boundary** ([`DataPlane::operate`]): giving KV cache segments
//!   back and creating them again, a sleep and a wake, putting loaded weights in place. A scheduler
//!   calls them once its batches in flight completed, and each reaches the device through its
//!   queue, so no running job shares a mapping change.
//! - **Loads beside the serving batches** ([`DataPlane::plan`], [`DataPlane::load`]): a checkpoint
//!   of the same model or another arrangement of its experts is published at the Storage Server
//!   while requests go on. The batches read the old weights until the boundary at which each new
//!   allocation is mapped at the address of the one it replaces.
//!
//! A weight is never written after it is published, by anyone. An update changes which published
//! allocation an address maps: the graphs captured over that address stay valid, the mapping stays
//! read-only, and the allocation is the one a relaunch on the new target finds under its own key.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use sha2::{Digest, Sha256};

use crate::batch::Submission;
use crate::completion::Outcome;
use crate::device::Executor;
use crate::memory::allocation::Backing as _;
use crate::memory::prefix_cache::PrefixCache;
use crate::memory::prefix_cache::planes::{CachePlanes, PlaneLayout};
use crate::memory::reload::{Matching, Reloading, Walk, WeightPair, paired};
use crate::memory::retained::Retained;
use crate::memory::weights::{LoadedWeight, WeightStore, retained_key};
use crate::{Error, Result, lock};

/// What a load of the served model reads: which checkpoint, and which expert sits in which slot.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Target {
    /// The checkpoint's directory.
    pub checkpoint: PathBuf,
    /// Each routed expert layer's logical expert ids in physical slot order, by module path. A
    /// layer this map does not name keeps the checkpoint's order.
    pub experts: BTreeMap<String, Vec<u32>>,
}

/// How a rank loads its model: the model crates are the executable's, so this crate reaches them
/// through what the executable hands it (§4.3).
pub trait ModelLoad: Send + Sync {
    /// Runs the model's load of `target` through `weights`, which every weight goes through.
    ///
    /// # Errors
    ///
    /// Why the checkpoint cannot be this model's, as a sentence.
    fn load(
        &self,
        target: &Target,
        weights: &mut dyn nn::RetainedWeights,
    ) -> std::result::Result<(), String>;

    /// The routed expert layers this rank's model holds, by module path, each with the experts its
    /// router names. A model without routed experts states none.
    fn expert_layers(&self) -> BTreeMap<String, u32>;
}

/// What the data plane holds of one rank.
pub struct Rank {
    /// The rank's device executor: the queue a change of its mappings goes through. Weak, because
    /// the epoch's close joins the device's thread and owns the last handle.
    pub executor: std::sync::Weak<Executor>,
    /// The rank's device.
    pub device: platform::DeviceId,
    /// The rank's Storage Server client.
    pub storage: Arc<Retained>,
    /// The rank's KV cache, on a rank that keeps one.
    pub cache: Option<Arc<Mutex<PrefixCache>>>,
    /// How the rank loads its model.
    pub model: Arc<dyn ModelLoad>,
    /// The weights the rank serves, in its load's order.
    pub serving: Vec<LoadedWeight>,
    /// The host the rank runs on, as a transfer manifest names it.
    pub host: String,
}

impl Rank {
    fn executor(&self) -> Result<Arc<Executor>> {
        self.executor
            .upgrade()
            .ok_or(Error::State("the rank's device has closed"))
    }
}

/// One rank's half of a load: what its model reads, and which weights move.
struct RankLoad {
    pairs: Vec<WeightPair>,
    /// The target's load, in order: the serving log once the load is in place.
    log: Vec<LoadedWeight>,
    /// Device bytes the replacement takes beside the weights that serve.
    bytes: u64,
}

/// A load the plan accepted, on every rank.
pub struct Planned {
    target: Target,
    ranks: Vec<RankLoad>,
    /// Whether the new weights are published yet, so a discard knows what there is to release.
    loaded: bool,
}

impl Planned {
    /// The most device bytes any rank needs beside what it serves.
    pub fn bytes(&self) -> u64 {
        self.ranks.iter().map(|rank| rank.bytes).max().unwrap_or(0)
    }

    /// How many weights move, over every rank.
    pub fn weights(&self) -> usize {
        self.ranks.iter().map(|rank| rank.pairs.len()).sum()
    }
}

/// The room one device has beside what it holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Room {
    /// Bytes the deployment's quota of the GPU leaves.
    pub quota: u64,
    /// Bytes the device has free.
    pub free: u64,
}

impl Room {
    /// Whether a load of `bytes` fits here once `given_back` bytes of the device leave. The graph
    /// pools a process gives back free the device and no quota, because the ledger books what the
    /// Storage Server retains and not what a process maps of its own.
    pub fn holds(self, bytes: u64, given_back: u64) -> bool {
        bytes <= self.quota.min(self.free.saturating_add(given_back))
    }
}

/// An expert redistribution between its plan and its commit.
enum Experts {
    /// The plan is made and the storage for its copies is being found.
    Planned(Planned),
    /// The copies run beside the serving batches.
    Copying(std::thread::JoinHandle<Result<Planned>>),
    /// The copies completed, and the new weights wait for the common boundary.
    Ready(Planned),
}

/// What a rank holds of a loaded LoRA adapter (§7.6): its tensors, published and mapped read-only.
struct Adapter {
    keys: Vec<[u8; 32]>,
    /// Each tensor's region, by its name in the adapter's file. Nothing reads them until a kernel
    /// program of the epoch does; a slot's tensors are what such a program is handed.
    regions: BTreeMap<String, nn::Backing>,
}

/// The data plane of one process: every rank it holds.
pub struct DataPlane {
    ranks: Vec<Mutex<Rank>>,
    /// What the ranks serve now.
    target: Mutex<Target>,
    /// A weight update's load, waiting for its boundary.
    staged: Mutex<Option<Planned>>,
    experts: Mutex<Option<Experts>>,
    /// Each rank's adapters, by the slot the daemon assigned.
    adapters: Mutex<BTreeMap<u32, Vec<Adapter>>>,
    /// Whether the graph pools are suspended, which only a wake or a load's end reverses.
    suspended: AtomicBool,
    /// A mapping change failed part of the way: the rank's weights are no one checkpoint's, and the
    /// epoch ends rather than computing with them.
    poisoned: Mutex<Option<String>>,
    /// The tokens the deployment fixed the cache at, when it fixed a capacity.
    max_total_tokens: Option<u64>,
    request_slots: u64,
}

fn rejected(what: &str, error: impl std::fmt::Display) -> Error {
    Error::Invalid(format!("{what}: {error}"))
}

/// Waits for a storage command of `executor`'s queue.
fn done(what: &str, submission: Submission) -> Result<()> {
    match submission.wait() {
        Outcome::Success(_) => Ok(()),
        Outcome::Failed(error) => Err(Error::Interface(format!("{what}: {error}"))),
        Outcome::Cancelled => Err(Error::Interface(format!("{what} was cancelled"))),
    }
}

/// What a rank's planes answer of the segments they hold: `{"held": count}`, `null` for planes
/// that are not split.
fn held(planes: Option<&Arc<CachePlanes>>) -> serde_json::Value {
    match planes {
        Some(planes) if planes.segments().is_some() => {
            serde_json::json!({ "held": planes.held_segments() })
        }
        _ => serde_json::Value::Null,
    }
}

/// The digest a set of weights is known by: what roots the prefix hash chains of the pages computed
/// with them, so a page of another checkpoint never matches.
fn identity_of(log: &[LoadedWeight]) -> String {
    let mut digest = Sha256::new();
    for weight in log {
        let text = weight.key.to_string();
        digest.update((text.len() as u64).to_le_bytes());
        digest.update(text.as_bytes());
    }
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

impl DataPlane {
    /// The data plane over `ranks`, which serve `target`. `max_total_tokens` is the capacity the
    /// deployment fixed for the cache, when it fixed one, and `request_slots` the rank's slots.
    pub fn new(
        ranks: Vec<Rank>,
        target: Target,
        max_total_tokens: Option<u64>,
        request_slots: u64,
    ) -> Self {
        Self {
            ranks: ranks.into_iter().map(Mutex::new).collect(),
            target: Mutex::new(target),
            staged: Mutex::new(None),
            experts: Mutex::new(None),
            adapters: Mutex::new(BTreeMap::new()),
            suspended: AtomicBool::new(false),
            poisoned: Mutex::new(None),
            max_total_tokens,
            request_slots,
        }
    }

    /// Why this epoch must end, once a change of its mappings failed part of the way.
    pub fn poisoned(&self) -> Option<String> {
        lock(&self.poisoned).clone()
    }

    /// Whether any slot holds an adapter.
    pub fn holds_adapters(&self) -> bool {
        !lock(&self.adapters).is_empty()
    }

    /// The target the ranks serve now.
    pub fn target(&self) -> Target {
        lock(&self.target).clone()
    }

    /// The hexadecimal key seed of each of `rank`'s cache groups for weights of `identity`, when it
    /// copies pages through a host KV cache; none when it keeps none, which keys nothing.
    fn page_contents(rank: &Rank, identity: &str) -> Vec<String> {
        let Some(cache) = rank.cache.as_ref().filter(|cache| lock(cache).copies_pages()) else {
            return Vec::new();
        };
        let plan = Arc::clone(lock(cache).planes());
        let plan = plan.plan();
        crate::memory::prefix_cache::host::page_contents(
            plan,
            &vec![identity.to_owned(); plan.groups.len()],
        )
        .iter()
        .map(|content| content.iter().map(|byte| format!("{byte:02x}")).collect())
        .collect()
    }

    /// What the rank's pages are computed from, which keys them in the host KV cache: the identity
    /// of the weights it serves. An epoch of the same weights finds the pages another left there,
    /// and one of other weights, or this one after a weight update, keys its own.
    ///
    /// # Errors
    ///
    /// When this data plane holds no rank.
    pub fn serving_identity(&self) -> Result<String> {
        let rank = self
            .ranks
            .first()
            .ok_or(Error::State("the data plane holds no rank"))?;
        Ok(identity_of(&lock(rank).serving))
    }

    fn planes_of(rank: &Rank) -> Option<Arc<CachePlanes>> {
        rank.cache
            .as_ref()
            .map(|cache| Arc::clone(lock(cache).planes()))
    }

    /// The room every device has for a load, in rank order.
    ///
    /// # Errors
    ///
    /// When the ledger or the device cannot be read.
    pub fn room(&self) -> Result<Vec<Room>> {
        self.ranks
            .iter()
            .map(|rank| {
                let rank = lock(rank);
                let device = rank.device;
                let (ledger, _) = rank.storage.ledger(device)?;
                let free = device.memory_info()?.free;
                Ok(Room {
                    quota: ledger
                        .quota
                        .map_or(free, |quota| quota.saturating_sub(ledger.used())),
                    free,
                })
            })
            .collect()
    }

    /// The bytes of device storage the graph pools hold, on the device that holds the fewest.
    pub fn graph_bytes(&self) -> u64 {
        self.ranks
            .iter()
            .filter_map(|rank| lock(rank).executor.upgrade())
            .map(|executor| executor.memory().graph)
            .min()
            .unwrap_or(0)
    }

    /// Plans a load of `target` on every rank, moving no byte.
    ///
    /// # Errors
    ///
    /// When the target cannot be the served model: the sentence names its first difference.
    pub fn plan(&self, target: Target, matching: Matching) -> Result<Planned> {
        let mut ranks = Vec::with_capacity(self.ranks.len());
        for rank in &self.ranks {
            let (model, serving, storage, device) = {
                let rank = lock(rank);
                (
                    Arc::clone(&rank.model),
                    rank.serving.clone(),
                    Arc::clone(&rank.storage),
                    rank.device,
                )
            };
            let mut walk = Walk::default();
            model
                .load(&target, &mut walk)
                .map_err(|error| Error::Invalid(error))?;
            let log = walk.into_log();
            let pairs = paired(&serving, &log, matching).map_err(|difference| {
                Error::Invalid(format!(
                    "{} loads other tensors than the served model: {difference}",
                    target.checkpoint.display()
                ))
            })?;
            let mut bytes = 0;
            for pair in &pairs {
                let storage = storage
                    .mapped_weight(device, &retained_key(&pair.old))?
                    .ok_or_else(|| {
                        Error::Invariant(format!("the rank maps no weight {}", pair.old))
                    })?;
                bytes += storage.bytes() as u64;
            }
            ranks.push(RankLoad { pairs, log, bytes });
        }
        Ok(Planned {
            target,
            ranks,
            loaded: false,
        })
    }

    /// Publishes what `planned` changes on every rank, each rank on a thread of its own: the
    /// batches go on reading the weights that serve.
    ///
    /// # Errors
    ///
    /// When a rank's load fails: everything this load published is released, and the ranks serve
    /// on as they did.
    pub fn load(&self, mut planned: Planned) -> Result<Planned> {
        let target = planned.target.clone();
        let loads: Vec<(Arc<dyn ModelLoad>, Arc<Retained>, platform::DeviceId, Vec<LoadedWeight>)> =
            self.ranks
                .iter()
                .map(|rank| {
                    let rank = lock(rank);
                    (
                        Arc::clone(&rank.model),
                        Arc::clone(&rank.storage),
                        rank.device,
                        rank.serving.clone(),
                    )
                })
                .collect();
        planned.loaded = true;
        let results: Vec<std::result::Result<Vec<LoadedWeight>, String>> =
            std::thread::scope(|scope| {
                let handles: Vec<_> = loads
                    .iter()
                    .map(|(model, storage, device, serving)| {
                        let target = &target;
                        scope.spawn(move || {
                            let mut store =
                                Reloading::new(Arc::clone(storage), *device, serving);
                            model.load(target, &mut store)?;
                            let (log, published, gathered) = store.finish();
                            tracing::info!(
                                device = device.0,
                                published,
                                gathered,
                                "a rank published what the load changes beside the weights that serve"
                            );
                            Ok(log)
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|handle| {
                        handle
                            .join()
                            .unwrap_or_else(|_| Err("a rank's weight load panicked".to_owned()))
                    })
                    .collect()
            });
        let mut failure = None;
        for (rank, result) in planned.ranks.iter().zip(results) {
            match result {
                // The walk and the load are one code path over one checkpoint, so a difference
                // here is a checkpoint that changed between them.
                Ok(log) if log == rank.log => {}
                Ok(_) => {
                    failure.get_or_insert_with(|| {
                        format!(
                            "{} changed while it loaded: its weights are not the ones planned",
                            target.checkpoint.display()
                        )
                    });
                }
                Err(error) => {
                    failure.get_or_insert(error);
                }
            }
        }
        if let Some(failure) = failure {
            self.discard(planned);
            return Err(Error::Invalid(format!(
                "the weights of {} did not load: {failure}",
                target.checkpoint.display()
            )));
        }
        Ok(planned)
    }

    /// Ends a load that will not serve: what it published goes back, and the ranks serve on.
    pub fn discard(&self, planned: Planned) {
        if !planned.loaded {
            return;
        }
        for (rank, load) in self.ranks.iter().zip(&planned.ranks) {
            let rank = lock(rank);
            let keys: Vec<[u8; 32]> = load
                .pairs
                .iter()
                .map(|pair| retained_key(&pair.new))
                .collect();
            if let Err(error) = rank.storage.forget_weights(rank.device, &keys) {
                tracing::warn!(%error, "an abandoned weight load's storage was not released");
            }
        }
    }

    /// Keeps a finished load for the boundary that puts it in place.
    pub fn stage(&self, planned: Planned) {
        if let Some(previous) = lock(&self.staged).replace(planned) {
            self.discard(previous);
        }
    }

    /// Ends a staged load that will not serve.
    pub fn discard_staged(&self) {
        if let Some(planned) = lock(&self.staged).take() {
            self.discard(planned);
        }
    }

    /// Maps `planned`'s weights at the addresses of the ones they replace, on every rank, at a
    /// boundary where nothing runs, and answers each rank's report.
    fn rebind(&self, planned: Planned) -> Result<Vec<String>> {
        let Planned { target, ranks, .. } = planned;
        let mut queued = Vec::with_capacity(ranks.len());
        for (rank, load) in self.ranks.iter().zip(&ranks) {
            let rank = lock(rank);
            let device = rank.device;
            let storage = Arc::clone(&rank.storage);
            let pairs: Vec<([u8; 32], [u8; 32])> = load
                .pairs
                .iter()
                .map(|pair| (retained_key(&pair.old), retained_key(&pair.new)))
                .collect();
            let work: Arc<dyn Fn() -> Result<()> + Send + Sync> = Arc::new(move || {
                storage.rebind_weights(device, &pairs).map(|moved| {
                    tracing::info!(device = device.0, moved, "rebound the rank's weights");
                })
            });
            queued.push(rank.executor()?.rebind(work)?);
        }
        let mut failure = None;
        for submission in queued {
            if let Err(error) = done("putting the new weights in place", submission) {
                failure.get_or_insert(error.to_string());
            }
        }
        if let Some(failure) = failure {
            *lock(&self.poisoned) = Some(failure.clone());
            return Err(Error::Interface(failure));
        }
        let mut reports = Vec::with_capacity(ranks.len());
        for (rank, load) in self.ranks.iter().zip(ranks) {
            let mut rank = lock(rank);
            let identity = identity_of(&load.log);
            let groups = Self::planes_of(&rank).map_or(0, |planes| planes.plan().groups.len());
            let moved = load.pairs.len();
            // The rank's pages in the host KV cache were computed from the weights this replaces,
            // so its later pages are keyed by the new ones: a prompt no longer finds the old.
            let page_contents = Self::page_contents(&rank, &identity);
            rank.serving = load.log;
            reports.push(
                serde_json::json!({
                    "group": rank.storage.scope().group,
                    "rank": rank.storage.scope().rank,
                    "identity": identity,
                    "identities": vec![identity.clone(); groups],
                    "page_contents": page_contents,
                    "moved": moved,
                })
                .to_string(),
            );
        }
        *lock(&self.target) = target;
        Ok(reports)
    }

    /// Suspends the graph pools of every rank, at a boundary where nothing runs: what a sleep gives
    /// back beside the cache, and what `keep` gives up for a load the free storage does not hold.
    ///
    /// # Errors
    ///
    /// When a device refuses; the pools suspended before stay so, and a restore brings them back.
    pub fn suspend_graphs(&self) -> Result<()> {
        if self.suspended.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        for rank in &self.ranks {
            let executor = lock(rank).executor()?;
            done("suspending the graph pools", executor.suspend_graphs()?)?;
        }
        tracing::info!("the graph pools gave their storage back at unchanged addresses");
        Ok(())
    }

    /// Restores what [`Self::suspend_graphs`] gave back, at the addresses the graphs hold.
    ///
    /// # Errors
    ///
    /// When a device refuses, naming the pool; nothing executes on the rank then.
    pub fn restore_graphs(&self) -> Result<()> {
        if !self.suspended.swap(false, Ordering::AcqRel) {
            return Ok(());
        }
        for rank in &self.ranks {
            let executor = lock(rank).executor()?;
            done("restoring the graph pools", executor.restore_graphs()?)?;
        }
        tracing::info!("the graph pools took their storage back at the addresses the graphs hold");
        Ok(())
    }

    /// What every rank reports of its cache's storage (`Scheduler::storage_reports` reads it).
    fn cache_storage(&self) -> Result<Vec<String>> {
        self.ranks
            .iter()
            .map(|rank| {
                let rank = lock(rank);
                let device = rank.device;
                let planes = Self::planes_of(&rank);
                let memory = device.memory_info()?;
                let graph_share = crate::capacity::graph_storage_budget_bytes(
                    i64::try_from(memory.total).unwrap_or(i64::MAX),
                );
                let (ledger, host_storage) = rank.storage.ledger(device)?;
                let scratch = rank.cache.as_ref().map_or(0, |cache| lock(cache).scratch_lent());
                // The head segment a successor creates beside this rank's: the sentinel, the
                // scratch of a warmup like this rank's and the writable pages one request needs to
                // advance, in whole granules.
                let head_bytes = match planes.as_ref() {
                    Some(planes) => {
                        let plan = planes.plan();
                        let granularity =
                            platform::HandleType::Descriptor.granularity(device)? as u64;
                        u64::from(PlaneLayout::whole_granules(
                            1 + scratch + plan.progress_units(),
                            plan.plane_bytes,
                            granularity,
                        )) * planes.unit_bytes()
                    }
                    None => 0,
                };
                let capacity_units = match (planes.as_ref(), self.max_total_tokens) {
                    (Some(planes), Some(tokens)) => {
                        Some(planes.plan().token_units(tokens, self.request_slots))
                    }
                    _ => None,
                };
                Ok(serde_json::json!({
                    "segments": planes.as_ref().and_then(|planes| planes.segments()).unwrap_or_default(),
                    "held": planes.as_ref().map_or(0, |planes| planes.held_segments()),
                    "unit_bytes": planes.as_ref().map_or(0, |planes| planes.unit_bytes()),
                    "head_bytes": head_bytes,
                    "capacity_units": capacity_units,
                    "scratch_units": scratch,
                    "free_bytes": memory.free,
                    "graph_share_bytes": graph_share,
                    "capturable": true,
                    "host": rank.host,
                    "scope": rank.storage.scope(),
                    "world": rank.storage.world(),
                    "ledger": ledger,
                    "host_storage": host_storage,
                })
                .to_string())
            })
            .collect()
    }

    /// Gives back the segments from `from` on, on every rank. No request holds a unit of them: the
    /// scheduler withdrew them and waited.
    fn release_cache(&self, from: u32) -> Result<Vec<String>> {
        let mut answers = Vec::with_capacity(self.ranks.len());
        for rank in &self.ranks {
            let rank = lock(rank);
            done(
                "giving the cache's segments back",
                rank.executor()?.release_cache(from)?,
            )?;
            answers.push(held(Self::planes_of(&rank).as_ref()).to_string());
        }
        Ok(answers)
    }

    /// Creates the first `limit` given-back segments again, or all of them, on every rank.
    fn restore_cache(&self, limit: Option<u32>) -> Result<Vec<String>> {
        let mut answers = Vec::with_capacity(self.ranks.len());
        for rank in &self.ranks {
            let rank = lock(rank);
            done(
                "creating the cache's segments again",
                rank.executor()?.wake(limit)?,
            )?;
            answers.push(held(Self::planes_of(&rank).as_ref()).to_string());
        }
        Ok(answers)
    }

    /// Gives back every segment and the graph storage, on every rank.
    fn sleep(&self) -> Result<Vec<String>> {
        self.suspend_graphs()?;
        let mut answers = Vec::with_capacity(self.ranks.len());
        for rank in &self.ranks {
            let rank = lock(rank);
            done("the sleep's release", rank.executor()?.sleep()?)?;
            answers.push(serde_json::Value::Null.to_string());
        }
        tracing::info!("the ranks sleep: their graph storage and cache are given back");
        Ok(answers)
    }

    /// Restores what [`Self::sleep`] gave back.
    fn wake(&self) -> Result<Vec<String>> {
        self.restore_graphs()?;
        self.restore_cache(None)
    }

    /// Runs the storage operation `name` with the JSON `payload` on every rank, at a boundary where
    /// no batch runs, and answers each rank's JSON result.
    ///
    /// # Errors
    ///
    /// When a rank cannot, and for a name no operation of this process answers to.
    pub fn operate(&self, name: &str, payload: &str) -> Result<Vec<String>> {
        let value: serde_json::Value = if payload.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::from_str(payload).map_err(|error| rejected("an operation's payload", error))?
        };
        match name {
            "uniserve.cache_storage" => self.cache_storage(),
            "uniserve.release_cache" => {
                let from = value["from"]
                    .as_u64()
                    .and_then(|from| u32::try_from(from).ok())
                    .ok_or_else(|| Error::Invalid("release_cache names no segment".into()))?;
                self.release_cache(from)
            }
            "uniserve.restore_cache" => self.restore_cache(
                value["segments"]
                    .as_u64()
                    .and_then(|count| u32::try_from(count).ok()),
            ),
            "uniserve.sleep" => self.sleep(),
            "uniserve.wake" => self.wake(),
            "uniserve.update_weights" => {
                let planned = lock(&self.staged).take().ok_or(Error::State(
                    "no weights are loaded for this update to put in place",
                ))?;
                self.rebind(planned)
            }
            other => Err(Error::Interface(format!(
                "this process has no storage operation named {other:?}"
            ))),
        }
    }

    /// Plans a redistribution of the routed experts (§7.6): checks `orders`, states what each rank
    /// needs, and holds the plan until its copies start.
    ///
    /// # Errors
    ///
    /// When an order is no permutation of its layer's experts, or names no routed layer; nothing
    /// changed.
    pub fn prepare_experts(&self, orders: BTreeMap<String, Vec<u32>>) -> Result<Vec<String>> {
        let mut experts = lock(&self.experts);
        if experts.is_some() {
            return Err(Error::State("an expert redistribution is already prepared"));
        }
        let layers = {
            let rank = self.ranks.first().ok_or(Error::State("no rank is served"))?;
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
        let planned = self.plan(target, Matching::Rearranged)?;
        let mut reports = Vec::with_capacity(self.ranks.len());
        for (rank, load) in self.ranks.iter().zip(&planned.ranks) {
            let rank = lock(rank);
            reports.push(
                serde_json::json!({
                    "orders": orders,
                    "gpu": rank.device.uuid()?.0,
                    "bytes": load.bytes,
                })
                .to_string(),
            );
        }
        *experts = Some(Experts::Planned(planned));
        Ok(reports)
    }

    /// Starts the copies of the prepared redistribution, beside the serving batches.
    ///
    /// # Errors
    ///
    /// When no redistribution is prepared, or it started already.
    pub fn start_experts(self: &Arc<Self>) -> Result<Vec<String>> {
        let mut experts = lock(&self.experts);
        let planned = match experts.take() {
            Some(Experts::Planned(planned)) => planned,
            other => {
                *experts = other;
                return Err(Error::State("no expert redistribution waits to start"));
            }
        };
        let plane = Arc::clone(self);
        let copying = std::thread::Builder::new()
            .name("expert-copies".into())
            .spawn(move || plane.load(planned))
            .map_err(|error| rejected("starting the expert copies", error))?;
        *experts = Some(Experts::Copying(copying));
        Ok(self.ranks.iter().map(|_| "null".to_owned()).collect())
    }

    /// Whether the copies completed, as each rank answers it.
    ///
    /// # Errors
    ///
    /// When a rank's copies failed; the redistribution is over, and the routing did not change.
    pub fn experts_ready(&self) -> Result<Vec<String>> {
        let mut experts = lock(&self.experts);
        let ready = match experts.take() {
            Some(Experts::Copying(task)) if task.is_finished() => {
                let planned = task
                    .join()
                    .map_err(|_| Error::Invariant("the expert copies panicked".into()))??;
                *experts = Some(Experts::Ready(planned));
                true
            }
            Some(Experts::Copying(task)) => {
                *experts = Some(Experts::Copying(task));
                false
            }
            Some(Experts::Ready(planned)) => {
                *experts = Some(Experts::Ready(planned));
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
    pub fn abort_experts(&self) -> Result<Vec<String>> {
        let state = lock(&self.experts).take();
        match state {
            Some(Experts::Planned(planned) | Experts::Ready(planned)) => self.discard(planned),
            Some(Experts::Copying(task)) => {
                // The copies own their allocations until they end, so they are waited for.
                if let Ok(planned) = task
                    .join()
                    .map_err(|_| Error::Invariant("the expert copies panicked".into()))?
                {
                    self.discard(planned);
                }
            }
            None => {}
        }
        Ok(self.ranks.iter().map(|_| "null".to_owned()).collect())
    }

    /// Puts the finished redistribution in place, at the boundary every scheduler met at.
    ///
    /// # Errors
    ///
    /// When its copies have not completed, or `order` is not the one they were made for.
    pub fn commit_experts(&self, order: &BTreeMap<String, Vec<u32>>) -> Result<Vec<String>> {
        let planned = match lock(&self.experts).take() {
            Some(Experts::Ready(planned)) => planned,
            other => {
                *lock(&self.experts) = other;
                return Err(Error::State("the expert copies have not completed"));
            }
        };
        if planned.target.experts != *order {
            let refusal = "the committed order is not the one the copies were made for";
            self.discard(planned);
            return Err(Error::Interface(refusal.into()));
        }
        self.rebind(planned)
    }

    /// Publishes the LoRA adapter `payload` names into its slot, on every rank (§7.6).
    ///
    /// The adapter's tensors take the path every weight takes: the Storage Server retains them under
    /// a key of their file and the rank maps them read-only, so a second engine of the deployment
    /// loading the same file shares them, and an unload returns them.
    ///
    /// # Errors
    ///
    /// When the payload names no path or slot, or the path holds no adapter; nothing stays loaded.
    pub fn load_adapter(&self, payload: &str) -> Result<Vec<String>> {
        let body: serde_json::Value =
            serde_json::from_str(payload).map_err(|error| rejected("an adapter's payload", error))?;
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
        let mut results = Vec::with_capacity(self.ranks.len());
        for rank in &self.ranks {
            let (storage, device) = {
                let rank = lock(rank);
                (Arc::clone(&rank.storage), rank.device)
            };
            let mut store = WeightStore::new(Arc::clone(&storage), device);
            let mut regions = BTreeMap::new();
            let mut loading = || -> Result<()> {
                let mut loader = nn::WeightLoader::new(&checkpoint, &mut store, None, None);
                for name in &names {
                    let shape = checkpoint
                        .tensor(name)
                        .ok_or_else(|| Error::Invariant(format!("no tensor {name}")))?
                        .shape;
                    let weight = loader
                        .load(&nn::weights::WeightRequest::region(name, 0, shape))
                        .map_err(|error| rejected("loading the adapter", error))?;
                    regions.insert(name.clone(), weight.backing);
                }
                Ok(())
            };
            let loaded = loading();
            let keys: Vec<[u8; 32]> = store
                .take_log()
                .iter()
                .map(|weight| retained_key(&weight.key))
                .collect();
            if let Err(error) = loaded {
                if let Err(cleanup) = storage.forget_weights(device, &keys) {
                    tracing::warn!(%cleanup, "a partial adapter load's storage was not released");
                }
                self.unload_slot(slot, &mut adapters);
                return Err(error);
            }
            results.push(
                serde_json::json!({ "adapter": slot, "tensors": regions.len() }).to_string(),
            );
            adapters.push(Adapter { keys, regions });
        }
        tracing::info!(slot, path, tensors = names.len(), "loaded an adapter's tensors");
        lock(&self.adapters).insert(slot, adapters);
        Ok(results)
    }

    fn unload_slot(&self, slot: u32, adapters: &mut Vec<Adapter>) {
        for (rank, adapter) in self.ranks.iter().zip(adapters.drain(..)) {
            let rank = lock(rank);
            if let Err(error) = rank
                .storage
                .forget_weights(rank.device, &adapter.keys)
            {
                tracing::warn!(%error, slot, "an adapter's storage was not released");
            }
            drop(adapter.regions);
        }
    }

    /// Releases the adapter of the slot `payload` names, on every rank.
    ///
    /// # Errors
    ///
    /// When the payload names no slot.
    pub fn unload_adapter(&self, payload: &str) -> Result<Vec<String>> {
        let body: serde_json::Value =
            serde_json::from_str(payload).map_err(|error| rejected("an adapter's payload", error))?;
        let slot = body["adapter"]
            .as_u64()
            .and_then(|slot| u32::try_from(slot).ok())
            .ok_or_else(|| Error::Invalid("unload_lora_adapter names no slot".into()))?;
        // A slot with no adapter is a load that failed part of the way, which the daemon cleans up
        // after: nothing is held, and nothing is owed.
        if let Some(mut adapters) = lock(&self.adapters).remove(&slot) {
            self.unload_slot(slot, &mut adapters);
        }
        Ok(self
            .ranks
            .iter()
            .map(|_| serde_json::json!({ "adapter": slot }).to_string())
            .collect())
    }
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
        // The graph storage gives back device storage and no quota.
        assert!(room.holds(100, 40));
        assert!(!room.holds(101, 1_000));
    }

    #[test]
    fn one_set_of_weights_has_one_identity_whatever_loads_it() {
        let weight = |tensor: &str| LoadedWeight {
            key: nn::WeightKey {
                files: Vec::new(),
                dtype: nn::DType::BF16,
                layout: nn::WeightLayout::ROW_MAJOR,
                region: nn::weights::WeightRegion {
                    tensors: vec![tensor.to_owned()],
                    start: 0,
                    shape: nn::Shape::new(&[1]),
                    stride: None,
                },
                provenance: None,
            },
            writable: false,
        };
        let first = identity_of(&[weight("a"), weight("b")]);
        assert_eq!(first, identity_of(&[weight("a"), weight("b")]));
        assert_ne!(first, identity_of(&[weight("a"), weight("c")]));
    }
}

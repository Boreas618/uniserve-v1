//! The host tier of video requests' latent trajectories, as the scheduler
//! drives it.
//!
//! A video request is preempted and resumed at denoising-step boundaries:
//! its trajectory's state at a step boundary is the samples of the bank
//! holding the committed step (the step count and the request's facts fix
//! everything else: an Euler step has no solver history, and every random
//! draw follows from the request's seed), which are derived state like a
//! KV cache page. Each denoising rank whose host runs a Storage Server maps
//! a host latent pool of its own, a page pool of pages the size of one page
//! of a request slot's latent banks (`ipc::LatentPoolInfo`), which the same
//! rank of every world of the deployment maps again. The scheduler decides
//! what moves:
//!
//! - **Offload.** A preempted request whose calls completed at step `k`
//!   (`0 < k <` its schedule) has every page of its slot that holds its
//!   samples reserved in its rank's pool under the page's key; the next
//!   batch to the denoiser's worker group, one of latent offloads alone,
//!   copies the pages of the committed bank into the reserved slots
//!   (`Batch::latent_offloads`), and its output commits them, or aborts
//!   them when it failed. A page the pool holds already (the same part of
//!   the same trajectory at the same step) needs no copy. The request then
//!   leaves with `k` stored steps and the record of where its pages lie and
//!   how each rank laid its part out ([`Record`]).
//! - **Restore.** A resumed request that names stored steps has the pages
//!   of every stored rank its record names pinned before its latent
//!   preparation is offered. When every page is pinned, every restoring
//!   rank lays the stored values out in its own pages through the copy
//!   manifest from the stored ranks' layouts to its own
//!   (`worker::storage::latent::restore`), copying a page it lays out as a
//!   stored rank did whole, and the preparation commits the trajectory at
//!   the stored step (`Call::latent_restore`); when one is missing, it
//!   starts from noise, which computes the same trajectory, and the pages
//!   pinned are released. The pins end once the preparation's result is
//!   accepted. The same path serves a request resumed by ranks of the
//!   parallel configuration that stored it and by ranks of another one.
//!
//! # Keys
//!
//! A page's key is the copy manifest library's key of the page as a chunk
//! of the rank's slot pages (`storage::manifest::Storage::paged`): the
//! digest of the logical regions of the samples its bytes hold and where
//! they lie in it, with the content of the values, `SHA-256(
//! "uniserve-latent-content" || seed || trajectory || steps)`: the latent
//! seed of what the deployment's denoising computes ([`latent_seed`]: the
//! weights' values and the function and format identities of the denoising
//! ranks' models), the trajectory's identity (`Trajectory::id`) and the
//! steps it took. The key is injective, since every input that determines a
//! page's values is in it, and minimal, since no input that does not is:
//! neither the world, the mesh, the rank nor the parallel configuration
//! that wrote the page, so a page two configurations lay out alike is one
//! page of the pool, and a world that computes other numerics finds none,
//! unless a world change keeps the state of the requests in flight
//! (`Trajectory::state_seed`). Keys are never reused for other bytes, so a
//! stale trajectory is never read: it leaves its pool least recently used
//! first.
//!
//! # Finding the pages of another layout
//!
//! A key names the layout the page was written in, which a restoring rank of
//! another parallel configuration does not know: the engine that stores a
//! trajectory records, for every stored rank, its pool, the host the pool
//! lies on and the rank's slot pages of the request as its executor
//! declared them in its latent preparation's completion
//! (`CallOutput::latent_layouts`). The record travels with the request's
//! facts through the daemon (`Trajectory::stored_layout`), as the stored
//! steps do; the engine that resumes the request computes the stored pages'
//! keys from it, pins them in the pools it names and hands every restoring
//! rank the stored ranks on its host (`LatentSource::readers`), which only
//! a rank of that host can map. A restoring rank reads stored pages of its
//! own host alone, so the restore hits only when each restoring rank's
//! host holds what the rank needs: the pages of a stored rank whose share
//! of every request is the rank's own (`LatentPoolInfo::region`), or else
//! the whole of every stored sample.
//!
//! The store connects as a head of its engine's world, so the trajectories
//! a world on trial stores are its own until its commit: a rollback discards
//! them, and the previous world computes the requests it takes back from
//! what they named before.
//!
//! Requests to the servers never block the scheduler: they are submitted,
//! and their answers are taken by [`TrajectoryStore::poll`] at every step.

use std::collections::{BTreeMap, HashMap};

use ipc::{
    LatentOffload, LatentPageCopy, LatentPoolInfo, LatentPoolRef, LatentRestore, LatentSource,
    RequestId, RequestKey,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use storage::manifest::region::{Span, atoms};
use storage::manifest::{Storage as PagedStorage, StorageLayout};
use storage::protocol::{PageSlot, Reply, Request};
use storage::{Key, Scope};

use crate::executor::WorkerGroupId;
use crate::host_tier::Servers;

/// The version of [`Record`]: a record of another version names no pages.
const RECORD_VERSION: u32 = 1;

/// One denoising rank's host latent pool.
#[derive(Clone)]
struct RankPool {
    /// The connection to the rank's host's server
    /// (`TrajectoryStore::servers`).
    client: usize,
    scope: Scope,
    pool: LatentPoolRef,
    /// What determines the rank's share of every request's samples.
    region: [u8; 32],
    /// The rank's host, and where an engine on another host reaches its
    /// server.
    host: String,
    endpoint: Option<String>,
}

/// One rank's report for [`TrajectoryStore::add_worker`].
pub(crate) struct RankLatents<'a> {
    /// The rank's host.
    pub host: &'a str,
    pub pool: &'a LatentPoolInfo,
}

/// Where a stored trajectory's pages lie and what each stored rank's pages
/// hold, as the engine that stored it records it with the request
/// (`Trajectory::stored_layout`, JSON).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Record {
    version: u32,
    /// The stored ranks, in rank order.
    ranks: Vec<StoredRank>,
}

/// One rank's part of a stored trajectory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct StoredRank {
    /// The host whose Storage Server holds the rank's pool, and where an
    /// engine on another host reaches that server.
    host: String,
    endpoint: Option<String>,
    pool: LatentPoolRef,
    /// What determines the rank's share of every request's samples
    /// (`LatentPoolInfo::region`).
    region: [u8; 32],
    /// The rank's slot pages of the request: the part of each sample it
    /// held and where its values lie, which keys its pages.
    layout: StorageLayout,
}

impl StoredRank {
    /// The pages that hold the rank's part of the samples.
    fn pages(&self) -> u64 {
        self.layout.bytes.div_ceil(self.pool.page_bytes.max(1))
    }
}

/// The latent seed of what a deployment's denoising computes, from the
/// identities its denoising ranks report (`LatentPoolInfo::identity`): the
/// first eight bytes of the SHA-256 of the distinct identities in order, so
/// that every build derives the same seed from the same identities, and a
/// parallel configuration of other ranks computing the same derives the
/// same seed. Zero when no rank reports one.
pub(crate) fn latent_seed<'a>(identities: impl IntoIterator<Item = &'a str>) -> u64 {
    let distinct: std::collections::BTreeSet<&str> = identities.into_iter().collect();
    if distinct.is_empty() {
        return 0;
    }
    let mut digest = Sha256::new();
    digest.update(b"uniserve-latent-seed\0");
    for identity in distinct {
        digest.update((identity.len() as u64).to_le_bytes());
        digest.update(identity.as_bytes());
    }
    let digest = digest.finalize();
    u64::from_le_bytes(digest[..8].try_into().unwrap_or_default())
}

/// The digest of what a trajectory's values are after `steps` steps: what
/// computed them (`seed`) and the trajectory.
fn content(seed: u64, trajectory: u64, steps: u32) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"uniserve-latent-content\0");
    digest.update(seed.to_le_bytes());
    digest.update(trajectory.to_le_bytes());
    digest.update(steps.to_le_bytes());
    digest.finalize().into()
}

/// The keys of the pages that hold a rank's part of a trajectory whose
/// values are `content`, laid out in `layout` on pages of `page_bytes`:
/// each page's key as a chunk of the rank's slot pages. `None` for a layout
/// that does not describe itself consistently.
fn page_keys(layout: &StorageLayout, page_bytes: u64, content: [u8; 32]) -> Option<Vec<Key>> {
    let paged = PagedStorage::paged(layout.clone(), page_bytes, &|_| Some(content));
    match paged {
        Ok(paged) => Some(paged.chunks.iter().map(|chunk| chunk.key).collect()),
        Err(error) => {
            tracing::warn!(%error, "a rank's latent layout keys no page");
            None
        }
    }
}

/// Whether the boxes `held` (one span per dimension each) hold every index
/// of a tensor of `shape`.
fn covers(shape: &[u64], held: &[Vec<Span>]) -> bool {
    // Cut each dimension where a box's span starts or ends: every product
    // of the pieces then lies wholly inside or outside each box.
    let pieces: Vec<Vec<Span>> = shape
        .iter()
        .enumerate()
        .map(|(dim, extent)| {
            let spans: Vec<Span> = held.iter().map(|spans| spans[dim]).collect();
            atoms(&Span::range(0, *extent), &spans)
        })
        .collect();
    let mut choice = vec![0_usize; shape.len()];
    loop {
        let cell: Vec<Span> = choice
            .iter()
            .enumerate()
            .map(|(dim, index)| pieces[dim][*index])
            .collect();
        let inside = held.iter().any(|spans| {
            spans
                .iter()
                .zip(&cell)
                .all(|(span, piece)| piece.within(span))
        });
        if !inside {
            return false;
        }
        // The next product of pieces, the last dimension fastest.
        let mut dim = shape.len();
        loop {
            if dim == 0 {
                return true;
            }
            dim -= 1;
            choice[dim] += 1;
            if choice[dim] < pieces[dim].len() {
                break;
            }
            choice[dim] = 0;
        }
    }
}

/// Whether the stored ranks `stored` hold every element of every sample
/// any stored rank holds part of.
fn hold_every_sample(stored: &[&StoredRank], all: &[StoredRank]) -> bool {
    let mut tensors: BTreeMap<&str, (&[u64], Vec<Vec<Span>>)> = BTreeMap::new();
    for rank in all {
        for field in &rank.layout.fields {
            tensors
                .entry(field.tensor.as_str())
                .or_insert((field.logical.shape.as_slice(), Vec::new()));
        }
    }
    for rank in stored {
        for field in &rank.layout.fields {
            let Some((_, held)) = tensors.get_mut(field.tensor.as_str()) else {
                continue;
            };
            // The local tensor holds the product of its dimensions'
            // segments.
            let mut boxes: Vec<Vec<Span>> = vec![Vec::new()];
            for segments in &field.logical.segments {
                boxes = boxes
                    .into_iter()
                    .flat_map(|prefix| {
                        segments.iter().map(move |segment| {
                            let mut spans = prefix.clone();
                            spans.push(segment.span);
                            spans
                        })
                    })
                    .collect();
            }
            held.extend(boxes);
        }
    }
    tensors
        .values()
        .all(|(shape, held)| !held.is_empty() && covers(shape, held))
}

/// What a submitted request is for.
enum Purpose {
    /// The reservation of an offload's pages on one rank.
    Reserve { request: RequestId, rank: usize },
    /// The commit of an offload's pages on one rank.
    Commit { request: RequestId },
    /// The pin of a restore's pages of one stored rank.
    Pin { request: RequestId, source: usize },
    /// A pin answered for a restore that ended meanwhile: whatever it
    /// pinned is released.
    Orphan { pin: Pinned, keys: Vec<Key> },
    /// A reservation answered for an offload that ended meanwhile: whatever
    /// it reserved is aborted.
    Abandoned { rank: RankPool, keys: Vec<Key> },
    /// An abort or an unpin, whose answer only reports failures.
    Settle,
}

/// Where an offload stands.
enum OffloadStage {
    /// Reservations submitted; each rank's slots once it answered.
    Reserving(Vec<Option<Vec<PageSlot>>>),
    /// Every rank reserved what it does not hold: the copies wait for a
    /// batch of the denoiser's worker group.
    Queued,
    /// The copies are carried by the submitted batch.
    Carried(u64),
    /// The copies completed and the commits are submitted: so many ranks
    /// have not answered.
    Committing(usize),
    /// The trajectory is stored.
    Stored,
    /// The trajectory is not stored: a rank's pool could not hold it, its
    /// copies failed, or a server was unreachable.
    Failed,
}

struct Offload {
    worker: WorkerGroupId,
    request_key: RequestKey,
    steps: u32,
    /// Every page's key, by rank.
    keys: Vec<Vec<Key>>,
    /// The pages each rank reserved, which the copies write and the commit
    /// or abort ends, by rank.
    reserved: Vec<Vec<(u32, u64)>>,
    /// The record of the stored trajectory, which the request leaves with.
    record: Record,
    stage: OffloadStage,
    /// Whether the request left while its copies were carried or
    /// committed: the offload settles its pool slots and is then dropped,
    /// as nobody asks how it ended.
    forgotten: bool,
}

/// A stored rank's pool, as a restore pins pages in it.
#[derive(Clone)]
struct Pinned {
    client: usize,
    pool: u64,
    /// The scope the pages are pinned for: the first restoring rank that
    /// reads them.
    reader: Scope,
}

/// One stored rank's pages a restore pins.
struct Source {
    pinned: Pinned,
    keys: Vec<Key>,
    /// What the restoring ranks are told of them.
    source: LatentSource,
}

/// Where a restore stands.
#[derive(PartialEq, Eq)]
enum RestoreStage {
    /// Pins submitted; each source's slots once it answered.
    Pinning(Vec<Option<Vec<Option<u64>>>>),
    /// Every page of every source is pinned.
    Hit(LatentRestore),
    /// A page is missing: the request starts from noise.
    Miss,
}

struct Restore {
    sources: Vec<Source>,
    steps: u32,
    content: [u8; 32],
    stage: RestoreStage,
}

/// How a preempted request's offload ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Offloaded {
    /// The trajectory is stored at this many steps, with the record of
    /// where its pages lie (`Trajectory::stored_layout`).
    Stored(u32, Vec<u8>),
    /// The trajectory is not stored.
    Failed,
}

/// How a resumed request's restore resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Restored {
    /// Every page is pinned: the latent preparation copies them.
    Hit(LatentRestore),
    /// A page is missing: the request starts from noise.
    Miss,
}

/// The scheduler's side of the host latent pools.
pub(crate) struct TrajectoryStore {
    /// The connections to the pools' servers, with the purpose of every
    /// request not yet answered.
    servers: Servers<Purpose>,
    /// The engine's host.
    host: String,
    /// The denoising ranks of each worker group the store serves, in rank
    /// order.
    workers: HashMap<WorkerGroupId, Vec<RankPool>>,
    /// The latent seed of what the scheduler's denoising computes.
    seed: u64,
    offloads: HashMap<RequestId, Offload>,
    restores: HashMap<RequestId, Restore>,
}

impl TrajectoryStore {
    /// Connect to the Storage Server of `host`, the engine's, at `socket`,
    /// as a head of `deployment` in `world`.
    pub(crate) fn connect(
        socket: &std::path::Path,
        deployment: &str,
        world: u64,
        host: &str,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            servers: Servers::connect(socket, deployment, world, "host latent pools")?,
            host: host.to_owned(),
            workers: HashMap::new(),
            seed: 0,
            offloads: HashMap::new(),
            restores: HashMap::new(),
        })
    }

    /// The connection to the server of a pool on `host`, reached at
    /// `endpoint` from another host.
    fn client_of(&mut self, host: &str, endpoint: Option<&str>) -> Option<usize> {
        let local = host == self.host;
        self.servers.client(local, endpoint)
    }

    /// Serve `worker`, whose denoising ranks (`ranks`, each with its rank
    /// index in the group) report host latent pools of equal slot pages,
    /// when every one's server is reachable. Returns whether the store
    /// serves it.
    pub(crate) fn add_worker(
        &mut self,
        worker: &WorkerGroupId,
        ranks: &[(u32, RankLatents<'_>)],
    ) -> bool {
        let Some((_, first)) = ranks.first() else {
            return false;
        };
        let slot_pages = first.pool.slot_pages;
        let mut pools = Vec::with_capacity(ranks.len());
        for (rank, report) in ranks {
            if report.pool.slot_pages != slot_pages || slot_pages == 0 {
                return false;
            }
            let Some(client) = self.client_of(report.host, report.pool.endpoint.as_deref()) else {
                return false;
            };
            pools.push(RankPool {
                client,
                scope: Scope {
                    deployment: self.servers.deployment().to_owned(),
                    group: worker.0.clone(),
                    rank: *rank,
                },
                pool: report.pool.pool,
                region: report.pool.region,
                host: report.host.to_owned(),
                endpoint: report.pool.endpoint.clone(),
            });
        }
        tracing::info!(
            worker = %worker.0,
            ranks = pools.len(),
            slot_pages,
            "the host latent pools serve the worker group"
        );
        self.workers.insert(worker.clone(), pools);
        true
    }

    /// Key trajectories from now on by `seed`, the latent seed of what the
    /// scheduler's denoising computes: one stored before keeps its old
    /// keys, which only a request that names that seed finds.
    pub(crate) fn reseed(&mut self, seed: u64) {
        self.seed = seed;
    }

    /// The latent seed trajectories are keyed by.
    pub(crate) fn seed(&self) -> u64 {
        self.seed
    }

    /// Whether the store serves `worker`.
    pub(crate) fn serves(&self, worker: &WorkerGroupId) -> bool {
        self.workers.contains_key(worker)
    }

    /// Start storing `request`'s trajectory `trajectory` after `steps` steps
    /// in the pools of `worker`'s denoising ranks, whose committed bank
    /// holds it (`request_key`'s slot) laid out as `layouts` state, each
    /// rank's slot pages of the request by its rank index
    /// (`CallOutput::latent_layouts`). Returns `false`, starting nothing,
    /// when the store does not serve `worker` or a rank's layout is missing
    /// or malformed; an offload a server could not be asked for fails
    /// ([`Self::offloaded`]) once the others answered.
    pub(crate) fn offload(
        &mut self,
        request: RequestId,
        request_key: RequestKey,
        worker: &WorkerGroupId,
        trajectory: u64,
        steps: u32,
        layouts: &BTreeMap<u32, String>,
    ) -> bool {
        let Some(ranks) = self.workers.get(worker).cloned() else {
            return false;
        };
        let content = content(self.seed, trajectory, steps);
        let mut keys = Vec::with_capacity(ranks.len());
        let mut stored = Vec::with_capacity(ranks.len());
        for rank in &ranks {
            let layout = layouts
                .get(&rank.scope.rank)
                .and_then(|layout| serde_json::from_str::<StorageLayout>(layout).ok());
            let Some(layout) = layout else {
                tracing::warn!(
                    request = request.0,
                    rank = rank.scope.rank,
                    "a denoising rank stated no latent layout of the request: its trajectory is \
                     not stored"
                );
                return false;
            };
            let Some(rank_keys) = page_keys(&layout, rank.pool.page_bytes, content) else {
                return false;
            };
            keys.push(rank_keys);
            stored.push(StoredRank {
                host: rank.host.clone(),
                endpoint: rank.endpoint.clone(),
                pool: rank.pool,
                region: rank.region,
                layout,
            });
        }
        let mut submitted = 0;
        for (index, (rank, keys)) in ranks.iter().zip(&keys).enumerate() {
            let reserve = Request::Reserve {
                pool: rank.pool.pool,
                writer: rank.scope.clone(),
                keys: keys.clone(),
                window: None,
            };
            if !self.servers.submit(
                rank.client,
                &reserve,
                Purpose::Reserve {
                    request,
                    rank: index,
                },
            ) {
                break;
            }
            submitted += 1;
        }
        let mut offload = Offload {
            worker: worker.clone(),
            request_key,
            steps,
            keys,
            reserved: vec![Vec::new(); ranks.len()],
            record: Record {
                version: RECORD_VERSION,
                ranks: stored,
            },
            stage: OffloadStage::Reserving(vec![None; ranks.len()]),
            forgotten: false,
        };
        if submitted < ranks.len() {
            // The ranks that were asked answer into an offload that failed:
            // their reservations are aborted as they arrive.
            offload.stage = OffloadStage::Failed;
        }
        self.offloads.insert(request, offload);
        true
    }

    /// The copies of `request`'s offload once every rank reserved its
    /// pages, for a batch of the denoiser's worker group with id `batch`,
    /// which carries them from now on; `None` while the reservations are
    /// answered, or once the copies were taken.
    pub(crate) fn take_offload(
        &mut self,
        request: RequestId,
        batch: u64,
    ) -> Option<(WorkerGroupId, LatentOffload)> {
        let offload = self.offloads.get_mut(&request)?;
        if !matches!(offload.stage, OffloadStage::Queued) {
            return None;
        }
        let ranks = self.workers.get(&offload.worker)?;
        let pages = offload
            .reserved
            .iter()
            .enumerate()
            .flat_map(|(index, reserved)| {
                let rank = &ranks[index];
                reserved.iter().map(move |&(page, slot)| LatentPageCopy {
                    rank: rank.scope.rank,
                    pool: rank.pool.pool,
                    slot,
                    page,
                })
            })
            .collect();
        offload.stage = OffloadStage::Carried(batch);
        Some((
            offload.worker.clone(),
            LatentOffload {
                request_key: offload.request_key,
                pages,
            },
        ))
    }

    /// The requests whose offload copies wait for a batch.
    pub(crate) fn queued(&self) -> Vec<RequestId> {
        self.offloads
            .iter()
            .filter(|(_, offload)| matches!(offload.stage, OffloadStage::Queued))
            .map(|(request, _)| *request)
            .collect()
    }

    /// How `request`'s offload ended, once it did; the offload is then
    /// forgotten. `None` while it runs, and for a request without one.
    pub(crate) fn offloaded(&mut self, request: RequestId) -> Option<Offloaded> {
        match self.offloads.get(&request)?.stage {
            OffloadStage::Stored => {}
            OffloadStage::Failed if !self.answering(request) => {
                self.offloads.remove(&request);
                return Some(Offloaded::Failed);
            }
            _ => return None,
        }
        let offload = self.offloads.remove(&request)?;
        match serde_json::to_vec(&offload.record) {
            Ok(record) => Some(Offloaded::Stored(offload.steps, record)),
            Err(error) => {
                tracing::warn!(%error, "a stored trajectory's record does not encode");
                Some(Offloaded::Failed)
            }
        }
    }

    /// Whether a reservation of `request`'s offload has not been answered.
    fn answering(&self, request: RequestId) -> bool {
        self.servers.pending().any(|purpose| {
            matches!(purpose, Purpose::Reserve { request: owner, .. } if *owner == request)
        })
    }

    /// The batch carrying offload copies completed: they are committed when
    /// it `succeeded`, and aborted otherwise.
    pub(crate) fn completed(&mut self, batch: u64, succeeded: bool) {
        let carried: Vec<RequestId> = self
            .offloads
            .iter()
            .filter(
                |(_, offload)| matches!(offload.stage, OffloadStage::Carried(id) if id == batch),
            )
            .map(|(request, _)| *request)
            .collect();
        for request in carried {
            if succeeded {
                self.commit(request);
            } else {
                tracing::warn!(
                    request = request.0,
                    "a trajectory's offload failed: it is not stored"
                );
                self.abort(request);
            }
        }
        self.sweep();
    }

    /// `worker` was lost: the offloads its batches carried did not happen.
    pub(crate) fn worker_failed(&mut self, worker: &WorkerGroupId) {
        let failed: Vec<RequestId> = self
            .offloads
            .iter()
            .filter(|(_, offload)| {
                offload.worker == *worker
                    && matches!(
                        offload.stage,
                        OffloadStage::Queued | OffloadStage::Carried(_)
                    )
            })
            .map(|(request, _)| *request)
            .collect();
        for request in failed {
            self.abort(request);
        }
    }

    /// Commit the pages `request`'s offload reserved, on every rank.
    fn commit(&mut self, request: RequestId) {
        let Some(offload) = self.offloads.get(&request) else {
            return;
        };
        let Some(ranks) = self.workers.get(&offload.worker).cloned() else {
            return;
        };
        let commits: Vec<(usize, Request)> = ranks
            .iter()
            .zip(&offload.reserved)
            .zip(&offload.keys)
            .filter(|((_, reserved), _)| !reserved.is_empty())
            .map(|((rank, reserved), keys)| {
                let keys = reserved
                    .iter()
                    .map(|&(page, _)| keys[page as usize])
                    .collect();
                (
                    rank.client,
                    Request::Commit {
                        pool: rank.pool.pool,
                        keys,
                    },
                )
            })
            .collect();
        let mut answering = 0;
        for (client, commit) in commits {
            if self
                .servers
                .submit(client, &commit, Purpose::Commit { request })
            {
                answering += 1;
            }
        }
        if let Some(offload) = self.offloads.get_mut(&request) {
            offload.stage = if answering == 0 {
                OffloadStage::Stored
            } else {
                OffloadStage::Committing(answering)
            };
        }
    }

    /// Abort the pages `request`'s offload reserved, on every rank: it
    /// failed.
    fn abort(&mut self, request: RequestId) {
        let Some(offload) = self.offloads.get_mut(&request) else {
            return;
        };
        offload.stage = OffloadStage::Failed;
        let reserved = std::mem::take(&mut offload.reserved);
        let keys = offload.keys.clone();
        let Some(ranks) = self.workers.get(&offload.worker).cloned() else {
            return;
        };
        for ((rank, reserved), keys) in ranks.iter().zip(reserved).zip(keys) {
            if reserved.is_empty() {
                continue;
            }
            let abort = Request::Abort {
                pool: rank.pool.pool,
                keys: reserved
                    .iter()
                    .map(|&(page, _)| keys[page as usize])
                    .collect(),
            };
            self.servers.submit(rank.client, &abort, Purpose::Settle);
        }
    }

    /// `rank` answered `request`'s reservation with `slots`: once every rank
    /// answered, the copies of what they reserved wait for a batch, or, when
    /// a pool could not hold a page, the reservations are aborted.
    fn reserved(&mut self, request: RequestId, rank: usize, slots: Vec<PageSlot>) {
        let Some(offload) = self.offloads.get_mut(&request) else {
            return;
        };
        let whole = slots.len() == offload.keys[rank].len()
            && slots.iter().all(|slot| !matches!(slot, PageSlot::Full));
        offload.reserved[rank] = slots
            .iter()
            .enumerate()
            .filter_map(|(page, slot)| match slot {
                PageSlot::Reserved { slot } => Some((page as u32, *slot)),
                PageSlot::Present | PageSlot::Full => None,
            })
            .collect();
        let answered = match &mut offload.stage {
            OffloadStage::Reserving(answers) => {
                answers[rank] = Some(slots);
                if !whole {
                    None
                } else {
                    Some(answers.iter().all(Option::is_some))
                }
            }
            // An offload that failed while this rank was asked aborts what
            // the rank reserved.
            _ => None,
        };
        match answered {
            Some(true) => {
                // A page every rank holds already needs no copy.
                offload.stage = if offload.reserved.iter().all(Vec::is_empty) {
                    OffloadStage::Stored
                } else {
                    OffloadStage::Queued
                };
            }
            Some(false) => {}
            None => {
                if let OffloadStage::Reserving(_) = offload.stage {
                    tracing::warn!(
                        request = request.0,
                        "a host latent pool cannot hold a preempted trajectory"
                    );
                }
                self.abort(request);
            }
        }
    }

    /// Pin `request`'s trajectory `trajectory` after `steps` steps, stored
    /// under `state_seed` or the store's own seed where `stored_layout` (the
    /// storing engine's record, `Trajectory::stored_layout`) says, for the
    /// denoising ranks of `worker` to lay out in their own pages. Returns
    /// `false`, starting nothing, when the store does not serve `worker`; a
    /// restore whose record names no pages, or whose pages lie where a
    /// restoring rank that needs them cannot read them, resolves as a miss.
    pub(crate) fn restore(
        &mut self,
        request: RequestId,
        worker: &WorkerGroupId,
        trajectory: u64,
        steps: u32,
        state_seed: Option<u64>,
        stored_layout: &[u8],
    ) -> bool {
        let Some(ranks) = self.workers.get(worker).cloned() else {
            return false;
        };
        let content = content(state_seed.unwrap_or(self.seed), trajectory, steps);
        let mut restore = Restore {
            sources: Vec::new(),
            steps,
            content,
            stage: RestoreStage::Miss,
        };
        match self.sources(request, &ranks, content, stored_layout) {
            Some(sources) => restore.sources = sources,
            None => {
                self.restores.insert(request, restore);
                return true;
            }
        }
        restore.stage = RestoreStage::Pinning(vec![None; restore.sources.len()]);
        for (index, source) in restore.sources.iter().enumerate() {
            let pin = Request::Pin {
                pool: source.pinned.pool,
                reader: source.pinned.reader.clone(),
                keys: source.keys.clone(),
            };
            if !self.servers.submit(
                source.pinned.client,
                &pin,
                Purpose::Pin {
                    request,
                    source: index,
                },
            ) {
                // A server that cannot be asked holds nothing for us; what
                // the others pin is released as they answer.
                restore.stage = RestoreStage::Miss;
                break;
            }
        }
        if restore.stage == RestoreStage::Miss {
            self.orphan_pins(request, &restore.sources);
        }
        self.restores.insert(request, restore);
        true
    }

    /// The stored ranks' pages `ranks`, the restoring ranks, read of the
    /// trajectory `stored_layout` records with values `content`: each
    /// stored rank on a restoring rank's host, with the keys of its pages
    /// and the restoring ranks of that host as its readers. `None` when the
    /// record names no pages, or a restoring rank's host holds less than
    /// the rank needs (module documentation).
    fn sources(
        &mut self,
        request: RequestId,
        ranks: &[RankPool],
        content: [u8; 32],
        stored_layout: &[u8],
    ) -> Option<Vec<Source>> {
        let record = match serde_json::from_slice::<Record>(stored_layout) {
            Ok(record) if record.version == RECORD_VERSION && !record.ranks.is_empty() => record,
            _ => {
                tracing::info!(
                    request = request.0,
                    "a resumed video request names no stored layout of this build"
                );
                return None;
            }
        };
        // Every restoring rank reads the stored pages of its own host: the
        // pages of a stored rank of its own share, or the whole samples.
        for rank in ranks {
            let local: Vec<&StoredRank> = record
                .ranks
                .iter()
                .filter(|stored| stored.host == rank.host)
                .collect();
            let shares = local.iter().any(|stored| stored.region == rank.region);
            if !shares && !hold_every_sample(&local, &record.ranks) {
                tracing::info!(
                    request = request.0,
                    rank = rank.scope.rank,
                    host = %rank.host,
                    "a stored trajectory's pages are not on the host of a rank that needs them"
                );
                return None;
            }
        }
        let mut sources = Vec::new();
        for stored in &record.ranks {
            let readers: Vec<&RankPool> = ranks
                .iter()
                .filter(|rank| rank.host == stored.host)
                .collect();
            let Some(first) = readers.first() else {
                continue;
            };
            if stored.pages() == 0 {
                continue;
            }
            let keys = page_keys(&stored.layout, stored.pool.page_bytes, content)?;
            let Ok(layout) = serde_json::to_string(&stored.layout) else {
                return None;
            };
            let client = self.client_of(&stored.host, stored.endpoint.as_deref())?;
            sources.push(Source {
                pinned: Pinned {
                    client,
                    pool: stored.pool.pool,
                    reader: first.scope.clone(),
                },
                keys,
                source: LatentSource {
                    readers: readers.iter().map(|rank| rank.scope.rank).collect(),
                    pool: stored.pool,
                    layout,
                    slots: Vec::new(),
                },
            });
        }
        (!sources.is_empty()).then_some(sources)
    }

    /// How `request`'s restore resolved; `None` while its pins are
    /// answered, and for a request without one.
    pub(crate) fn restored(&self, request: RequestId) -> Option<Restored> {
        match &self.restores.get(&request)?.stage {
            RestoreStage::Pinning(_) => None,
            RestoreStage::Hit(restore) => Some(Restored::Hit(restore.clone())),
            RestoreStage::Miss => Some(Restored::Miss),
        }
    }

    /// End `request`'s restore: the pins it holds are released. A restore
    /// whose pins are still answered releases them as they arrive.
    pub(crate) fn release(&mut self, request: RequestId) {
        let Some(restore) = self.restores.remove(&request) else {
            return;
        };
        match restore.stage {
            RestoreStage::Hit(_) => {
                for source in &restore.sources {
                    self.unpin(&source.pinned, source.keys.clone());
                }
            }
            RestoreStage::Pinning(answers) => {
                // The sources that answered release what they pinned now;
                // the others release it when they answer.
                for (source, answer) in restore.sources.iter().zip(answers) {
                    if let Some(slots) = answer {
                        self.unpin(&source.pinned, pinned(&source.keys, &slots));
                    }
                }
                self.orphan_pins(request, &restore.sources);
            }
            RestoreStage::Miss => {
                self.orphan_pins(request, &restore.sources);
            }
        }
    }

    /// Turn the unanswered pins of `request` into orphans, which release
    /// what they pinned when they are answered.
    fn orphan_pins(&mut self, request: RequestId, sources: &[Source]) {
        for purpose in self.servers.pending_mut() {
            if let Purpose::Pin {
                request: owner,
                source,
            } = purpose
                && *owner == request
            {
                *purpose = Purpose::Orphan {
                    pin: sources[*source].pinned.clone(),
                    keys: sources[*source].keys.clone(),
                };
            }
        }
    }

    /// Forget `request`, which left the scheduler: its restore's pins end,
    /// an offload that has not reached a batch is aborted, and one whose
    /// copies a batch carries settles its pool slots before it is dropped.
    pub(crate) fn forget(&mut self, request: RequestId) {
        self.release(request);
        if let Some(offload) = self.offloads.get_mut(&request)
            && matches!(
                offload.stage,
                OffloadStage::Carried(_) | OffloadStage::Committing(_)
            )
        {
            offload.forgotten = true;
            return;
        }
        if let Some(offload) = self.offloads.get(&request)
            && matches!(
                offload.stage,
                OffloadStage::Reserving(_) | OffloadStage::Queued
            )
        {
            let Some(ranks) = self.workers.get(&offload.worker).cloned() else {
                return;
            };
            let keys = offload.keys.clone();
            for purpose in self.servers.pending_mut() {
                if let Purpose::Reserve {
                    request: owner,
                    rank,
                } = purpose
                    && *owner == request
                {
                    *purpose = Purpose::Abandoned {
                        rank: ranks[*rank].clone(),
                        keys: keys[*rank].clone(),
                    };
                }
            }
            self.abort(request);
            self.offloads.remove(&request);
        }
    }

    fn unpin(&mut self, pin: &Pinned, keys: Vec<Key>) {
        if keys.is_empty() {
            return;
        }
        let unpin = Request::Unpin {
            pool: pin.pool,
            reader: pin.reader.clone(),
            keys,
        };
        self.servers.submit(pin.client, &unpin, Purpose::Settle);
    }

    /// Source `source` answered `request`'s pin with `slots`: once every
    /// source answered, the restore is a hit when every page is pinned, and
    /// a miss otherwise, which releases the pages pinned.
    fn pinned(&mut self, request: RequestId, source: usize, slots: Vec<Option<u64>>) {
        let Some(restore) = self.restores.get_mut(&request) else {
            return;
        };
        let RestoreStage::Pinning(answers) = &mut restore.stage else {
            return;
        };
        answers[source] = Some(slots);
        if !answers.iter().all(Option::is_some) {
            return;
        }
        let answers: Vec<Vec<Option<u64>>> = answers.iter_mut().filter_map(Option::take).collect();
        let whole = answers.iter().zip(&restore.sources).all(|(slots, source)| {
            slots.len() == source.keys.len() && slots.iter().all(Option::is_some)
        });
        if whole {
            let sources = restore
                .sources
                .iter()
                .zip(&answers)
                .map(|(source, slots)| LatentSource {
                    slots: slots.iter().flatten().copied().collect(),
                    ..source.source.clone()
                })
                .collect();
            restore.stage = RestoreStage::Hit(LatentRestore {
                steps: restore.steps,
                content: restore.content,
                sources,
            });
            return;
        }
        restore.stage = RestoreStage::Miss;
        let release: Vec<(Pinned, Vec<Key>)> = restore
            .sources
            .iter()
            .zip(&answers)
            .map(|(source, slots)| (source.pinned.clone(), pinned(&source.keys, slots)))
            .collect();
        for (pin, keys) in release {
            self.unpin(&pin, keys);
        }
    }

    /// Take the servers' answers. Returns whether a restore or an offload
    /// moved on.
    pub(crate) fn poll(&mut self) -> bool {
        let mut moved = false;
        for (purpose, reply) in self.servers.answers() {
            moved = true;
            match (purpose, reply) {
                (Purpose::Reserve { request, rank }, reply) => {
                    let slots = match reply {
                        Reply::Reserved { slots } => slots,
                        other => {
                            tracing::warn!(?other, "a host latent pool reservation failed");
                            Vec::new()
                        }
                    };
                    self.reserved(request, rank, slots);
                }
                (Purpose::Commit { request }, reply) => {
                    let committed = matches!(reply, Reply::Committed);
                    if !committed {
                        tracing::warn!(?reply, "a host latent pool commit failed");
                    }
                    if let Some(offload) = self.offloads.get_mut(&request)
                        && let OffloadStage::Committing(remaining) = &mut offload.stage
                    {
                        *remaining -= 1;
                        if !committed {
                            offload.stage = OffloadStage::Failed;
                        } else if *remaining == 0 {
                            offload.stage = OffloadStage::Stored;
                        }
                    }
                }
                (Purpose::Pin { request, source }, reply) => {
                    let slots = match reply {
                        Reply::Pinned { slots } => slots,
                        other => {
                            tracing::warn!(?other, "a host latent pool pin failed");
                            Vec::new()
                        }
                    };
                    self.pinned(request, source, slots);
                }
                (Purpose::Orphan { pin, keys }, reply) => {
                    if let Reply::Pinned { slots } = reply {
                        self.unpin(&pin, pinned(&keys, &slots));
                    }
                }
                (Purpose::Abandoned { rank, keys }, reply) => {
                    if let Reply::Reserved { slots } = reply {
                        let reserved: Vec<Key> = keys
                            .iter()
                            .zip(&slots)
                            .filter(|(_, slot)| matches!(slot, PageSlot::Reserved { .. }))
                            .map(|(key, _)| *key)
                            .collect();
                        if !reserved.is_empty() {
                            let abort = Request::Abort {
                                pool: rank.pool.pool,
                                keys: reserved,
                            };
                            self.servers.submit(rank.client, &abort, Purpose::Settle);
                        }
                    }
                }
                (Purpose::Settle, Reply::Error { kind, message }) => {
                    tracing::warn!(?kind, message, "a host latent pool request failed");
                }
                (Purpose::Settle, _) => {}
            }
        }
        self.sweep();
        moved
    }

    /// Drop the forgotten offloads that settled their pool slots.
    fn sweep(&mut self) {
        self.offloads.retain(|_, offload| {
            !(offload.forgotten
                && matches!(offload.stage, OffloadStage::Stored | OffloadStage::Failed))
        });
    }

    /// Whether a request to a server awaits its answer.
    pub(crate) fn waiting(&self) -> bool {
        self.servers.waiting()
    }
}

/// The keys of `keys` that `slots` pinned.
fn pinned(keys: &[Key], slots: &[Option<u64>]) -> Vec<Key> {
    keys.iter()
        .zip(slots)
        .filter(|(_, slot)| slot.is_some())
        .map(|(key, _)| *key)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two sequence-parallel ranks' rows of a 40-row sample, or one rank's.
    fn ranks(shares: &[(u64, u64)], hosts: &[&str]) -> Vec<StoredRank> {
        use storage::manifest::{Encoding, Field, Logical, Segment};
        shares
            .iter()
            .zip(hosts)
            .map(|(&(start, count), host)| StoredRank {
                host: (*host).to_owned(),
                endpoint: None,
                pool: LatentPoolRef {
                    pool: 1,
                    shape: [0; 32],
                    numa_node: None,
                    page_bytes: 256,
                },
                region: [start as u8; 32],
                layout: StorageLayout {
                    bytes: count * 16,
                    fields: vec![Field {
                        tensor: "video_latents".into(),
                        source: String::new(),
                        dtype: "float32".into(),
                        element: 4,
                        offset: 0,
                        shape: vec![count, 4],
                        strides: vec![4, 1],
                        logical: Logical {
                            shape: vec![40, 4],
                            dims: vec!["row".into(), "channel".into()],
                            units: vec![1, 0],
                            segments: vec![
                                vec![Segment {
                                    span: Span::range(start, count),
                                    local: 0,
                                }],
                                vec![Segment {
                                    span: Span::range(0, 4),
                                    local: 0,
                                }],
                            ],
                        },
                        encoding: Encoding::Affine,
                    }],
                },
            })
            .collect()
    }

    /// The ranks of one host hold every sample when their rows cover it,
    /// and not when another host holds some.
    #[test]
    fn a_host_holds_every_sample_when_its_ranks_cover_it() {
        let both = ranks(&[(0, 22), (22, 18)], &["a", "a"]);
        assert!(hold_every_sample(&both.iter().collect::<Vec<_>>(), &both));
        let split = ranks(&[(0, 22), (22, 18)], &["a", "b"]);
        let on_a: Vec<&StoredRank> = split.iter().filter(|rank| rank.host == "a").collect();
        assert!(!hold_every_sample(&on_a, &split));
        let overlapping = ranks(&[(0, 30), (10, 30)], &["a", "a"]);
        assert!(hold_every_sample(
            &overlapping.iter().collect::<Vec<_>>(),
            &overlapping
        ));
    }

    /// The latent seed of a configuration of other ranks computing the same
    /// is the same; other numerics have another.
    #[test]
    fn the_latent_seed_follows_what_the_ranks_compute() {
        assert_eq!(latent_seed(["a", "a"]), latent_seed(["a"]));
        assert_ne!(latent_seed(["a"]), latent_seed(["b"]));
        assert_eq!(latent_seed(std::iter::empty()), 0);
    }

    /// A record survives its encoding and keys a stored rank's pages by
    /// what they hold: the first page of rows a rank of either
    /// configuration lays out alike has one key.
    #[test]
    fn stored_pages_are_keyed_by_what_they_hold() {
        let two = ranks(&[(0, 22), (22, 18)], &["a", "a"]);
        let one = ranks(&[(0, 40)], &["a"]);
        let content = content(7, 77, 3);
        let first = page_keys(&two[0].layout, 256, content).unwrap();
        let whole = page_keys(&one[0].layout, 256, content).unwrap();
        assert_eq!(first.len(), 2);
        assert_eq!(whole.len(), 3);
        assert_eq!(first[0], whole[0]);
        assert_ne!(first[1], whole[1]);
        let record = Record {
            version: RECORD_VERSION,
            ranks: two,
        };
        let encoded = serde_json::to_vec(&record).unwrap();
        assert_eq!(serde_json::from_slice::<Record>(&encoded).unwrap(), record);
    }
}

//! The host tier of video requests' latent trajectories, as the scheduler
//! drives it.
//!
//! A video request is preempted and resumed at denoising-step boundaries:
//! its trajectory's state at a step boundary is the samples of the bank
//! holding the committed step (the step count and the request's facts fix
//! everything else: an Euler step has no solver history, and every random
//! draw follows from the request's seed), which are derived state like a
//! KV cache page. Each denoising rank whose host runs a Storage Server maps
//! a host latent pool, a page pool of pages the size of one page of a
//! request slot's latent banks (`ipc::LatentPoolInfo`), shared by the
//! deployment's denoising ranks on that host's NUMA node and by every world
//! of the deployment. The scheduler decides what moves:
//!
//! - **Offload.** A preempted request whose calls completed at step `k`
//!   (`0 < k <` its schedule) has every page of its slot reserved in every
//!   denoising rank's pool under the page's key; the next batch to the
//!   denoiser's worker group, one of latent offloads alone, copies the pages
//!   of the committed bank into the reserved slots (`Batch::latent_offloads`),
//!   and its output commits them, or aborts them when it failed. A page the
//!   pool holds already (the same trajectory stored at the same step)
//!   needs no copy. The request then leaves with `k` stored steps.
//! - **Restore.** A resumed request that names stored steps has every page
//!   key pinned in every rank's pool before its latent preparation is
//!   offered. When every page of every rank is pinned, the preparation
//!   copies them into the bank it initializes and commits the trajectory at
//!   the stored step (`Call::latent_restore`); when one is missing, it
//!   starts from noise, which computes the same trajectory, and the pages
//!   pinned are released. The pins end once the preparation's result is
//!   accepted.
//!
//! A page's key is `SHA-256("uniserve-latent-page" || content || trajectory
//! || steps || page)`, its content `SHA-256("uniserve-latent-content" ||
//! seed || region)`: the trajectory's identity (`Trajectory::id`), the steps
//! it took, the page's index in the slot, the latent seed of what the
//! deployment's denoising computes ([`latent_seed`]: the weights' values
//! and the function and format identities of the denoising ranks' models)
//! and the rank's region (the page geometry and element type, the video
//! plan that selects a request's layout, and the rank's share of the
//! denoiser's rows). A world whose denoising computes the same thing on the
//! same ranks thus finds the trajectories another stored, and one that
//! computes other numerics finds none, unless a world change keeps the
//! state of the requests in flight (`Trajectory::state_seed`). Keys are
//! never reused for other bytes, so a stale trajectory is never read: it
//! leaves its pool least recently used first.
//!
//! The store connects as a head of its engine's world, so the trajectories
//! a world on trial stores are its own until its commit: a rollback discards
//! them, and the previous world computes the requests it takes back from
//! what they named before.
//!
//! Requests to the servers never block the scheduler: they are submitted,
//! and their answers are taken by [`TrajectoryStore::poll`] at every step.

use std::collections::HashMap;

use ipc::{LatentOffload, LatentPageCopy, LatentPoolInfo, LatentRestore, RequestId, RequestKey};
use sha2::{Digest, Sha256};
use storage::protocol::{PageSlot, Reply, Request};
use storage::{Client, Key, Role, Scope};

use crate::executor::WorkerGroupId;

/// One denoising rank's host latent pool.
#[derive(Clone)]
struct RankPool {
    /// The connection to the rank's host's server, in
    /// `TrajectoryStore::clients`.
    client: usize,
    scope: Scope,
    pool: u64,
    /// The rank's region, which keys its pages with the latent seed.
    region: [u8; 32],
    /// The pages one request slot owns on the rank.
    slot_pages: u32,
}

/// One rank's report for [`TrajectoryStore::add_worker`].
pub(crate) struct RankLatents<'a> {
    /// Whether the rank runs on the engine's host.
    pub local: bool,
    pub pool: &'a LatentPoolInfo,
}

/// The latent seed of what a deployment's denoising computes, from the
/// identities its denoising ranks report (`LatentPoolInfo::identity`), in
/// rank order: the first eight bytes of the SHA-256 of the identities, so
/// that every build derives the same seed from the same identities. Zero
/// when no rank reports one.
pub(crate) fn latent_seed<'a>(identities: impl IntoIterator<Item = &'a str>) -> u64 {
    let mut digest = Sha256::new();
    digest.update(b"uniserve-latent-seed\0");
    let mut any = false;
    for identity in identities {
        any = true;
        digest.update((identity.len() as u64).to_le_bytes());
        digest.update(identity.as_bytes());
    }
    if !any {
        return 0;
    }
    let digest = digest.finalize();
    u64::from_le_bytes(digest[..8].try_into().unwrap_or_default())
}

/// The digest that seeds a rank's page keys: the latent seed of what
/// computed the trajectory and the rank's region.
fn content(seed: u64, region: &[u8; 32]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"uniserve-latent-content\0");
    digest.update(seed.to_le_bytes());
    digest.update(region);
    digest.finalize().into()
}

/// The key of page `page` of trajectory `trajectory` after `steps` steps
/// on a rank of `content`.
fn page_key(content: &[u8; 32], trajectory: u64, steps: u32, page: u32) -> Key {
    let mut digest = Sha256::new();
    digest.update(b"uniserve-latent-page\0");
    digest.update(content);
    digest.update(trajectory.to_le_bytes());
    digest.update(steps.to_le_bytes());
    digest.update(page.to_le_bytes());
    Key(digest.finalize().into())
}

/// The keys of every page of a slot on `rank`.
fn rank_keys(rank: &RankPool, seed: u64, trajectory: u64, steps: u32) -> Vec<Key> {
    let content = content(seed, &rank.region);
    (0..rank.slot_pages)
        .map(|page| page_key(&content, trajectory, steps, page))
        .collect()
}

/// What a submitted request is for.
enum Purpose {
    /// The reservation of an offload's pages on one rank.
    Reserve { request: RequestId, rank: usize },
    /// The commit of an offload's pages on one rank.
    Commit { request: RequestId },
    /// The pin of a restore's pages on one rank.
    Pin { request: RequestId, rank: usize },
    /// A pin answered for a restore that ended meanwhile: whatever it
    /// pinned is released.
    Orphan { rank: RankPool, keys: Vec<Key> },
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
    stage: OffloadStage,
    /// Whether the request left while its copies were carried or
    /// committed: the offload settles its pool slots and is then dropped,
    /// as nobody asks how it ended.
    forgotten: bool,
}

/// Where a restore stands.
#[derive(PartialEq, Eq)]
enum RestoreStage {
    /// Pins submitted; each rank's slots once it answered.
    Pinning(Vec<Option<Vec<Option<u64>>>>),
    /// Every page of every rank is pinned.
    Hit(LatentRestore),
    /// A page is missing: the request starts from noise.
    Miss,
}

struct Restore {
    worker: WorkerGroupId,
    /// Every page's key, by rank.
    keys: Vec<Vec<Key>>,
    steps: u32,
    stage: RestoreStage,
}

/// How a preempted request's offload ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Offloaded {
    /// The trajectory is stored at this many steps.
    Stored(u32),
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
    deployment: String,
    /// The world of the scheduler's engine: the servers keep what a world
    /// on trial commits its own until its commit.
    world: u64,
    /// The connections to the servers: the engine's host's first, then other
    /// hosts' by the TCP address they were reached at.
    clients: Vec<Client>,
    remote: HashMap<String, usize>,
    /// The denoising ranks of each worker group the store serves, in rank
    /// order.
    workers: HashMap<WorkerGroupId, Vec<RankPool>>,
    /// The latent seed of what the scheduler's denoising computes.
    seed: u64,
    /// Requests submitted and not yet answered, by connection and request.
    pending: HashMap<(usize, u64), Purpose>,
    offloads: HashMap<RequestId, Offload>,
    restores: HashMap<RequestId, Restore>,
}

impl TrajectoryStore {
    /// Connect to this host's Storage Server, at `socket`, as a head of
    /// `deployment` in `world`.
    pub(crate) fn connect(
        socket: &std::path::Path,
        deployment: &str,
        world: u64,
    ) -> anyhow::Result<Self> {
        let client = Client::connect(
            socket,
            Role::Head {
                deployment: deployment.to_owned(),
                world,
            },
        )?;
        Ok(Self {
            deployment: deployment.to_owned(),
            world,
            clients: vec![client],
            remote: HashMap::new(),
            workers: HashMap::new(),
            seed: 0,
            pending: HashMap::new(),
            offloads: HashMap::new(),
            restores: HashMap::new(),
        })
    }

    /// The connection to a remote server at `address`, made when absent.
    fn remote(&mut self, address: &str) -> Option<usize> {
        if let Some(&client) = self.remote.get(address) {
            return Some(client);
        }
        let role = Role::Head {
            deployment: self.deployment.clone(),
            world: self.world,
        };
        match Client::connect_tcp(address, role) {
            Ok(client) => {
                self.clients.push(client);
                self.remote
                    .insert(address.to_owned(), self.clients.len() - 1);
                Some(self.clients.len() - 1)
            }
            Err(error) => {
                tracing::warn!(address, %error, "the host latent pools cannot reach a storage server");
                None
            }
        }
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
            let client = if report.local {
                0
            } else {
                let Some(client) = report
                    .pool
                    .endpoint
                    .as_deref()
                    .and_then(|address| self.remote(address))
                else {
                    return false;
                };
                client
            };
            pools.push(RankPool {
                client,
                scope: Scope {
                    deployment: self.deployment.clone(),
                    group: worker.0.clone(),
                    rank: *rank,
                },
                pool: report.pool.pool,
                region: report.pool.region,
                slot_pages,
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

    fn submit(&mut self, client: usize, request: &Request, purpose: Purpose) -> bool {
        match self.clients[client].submit(request, &[]) {
            Ok(id) => {
                self.pending.insert((client, id), purpose);
                true
            }
            Err(error) => {
                tracing::warn!(%error, "a host latent pool request failed");
                false
            }
        }
    }

    /// Start storing `request`'s trajectory `trajectory` after `steps` steps
    /// in the pools of `worker`'s denoising ranks, whose committed bank
    /// holds it (`request_key`'s slot). Returns `false`, starting nothing,
    /// when the store does not serve `worker`; an offload a server could not
    /// be asked for fails ([`Self::offloaded`]) once the others answered.
    pub(crate) fn offload(
        &mut self,
        request: RequestId,
        request_key: RequestKey,
        worker: &WorkerGroupId,
        trajectory: u64,
        steps: u32,
    ) -> bool {
        let Some(ranks) = self.workers.get(worker).cloned() else {
            return false;
        };
        let keys: Vec<Vec<Key>> = ranks
            .iter()
            .map(|rank| rank_keys(rank, self.seed, trajectory, steps))
            .collect();
        let mut submitted = 0;
        for (index, (rank, keys)) in ranks.iter().zip(&keys).enumerate() {
            let reserve = Request::Reserve {
                pool: rank.pool,
                writer: rank.scope.clone(),
                keys: keys.clone(),
                window: None,
            };
            if !self.submit(
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
                    pool: rank.pool,
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
        let outcome = match self.offloads.get(&request)?.stage {
            OffloadStage::Stored => Offloaded::Stored(self.offloads[&request].steps),
            OffloadStage::Failed if !self.answering(request) => Offloaded::Failed,
            _ => return None,
        };
        self.offloads.remove(&request);
        Some(outcome)
    }

    /// Whether a reservation of `request`'s offload has not been answered.
    fn answering(&self, request: RequestId) -> bool {
        self.pending.values().any(|purpose| {
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
                        pool: rank.pool,
                        keys,
                    },
                )
            })
            .collect();
        let mut answering = 0;
        for (client, commit) in commits {
            if self.submit(client, &commit, Purpose::Commit { request }) {
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
                pool: rank.pool,
                keys: reserved
                    .iter()
                    .map(|&(page, _)| keys[page as usize])
                    .collect(),
            };
            self.submit(rank.client, &abort, Purpose::Settle);
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
    /// under `state_seed` or the store's own seed, in the pools of
    /// `worker`'s denoising ranks. Returns `false`, starting nothing, when
    /// the store does not serve `worker`.
    pub(crate) fn restore(
        &mut self,
        request: RequestId,
        worker: &WorkerGroupId,
        trajectory: u64,
        steps: u32,
        state_seed: Option<u64>,
    ) -> bool {
        let Some(ranks) = self.workers.get(worker).cloned() else {
            return false;
        };
        let seed = state_seed.unwrap_or(self.seed);
        let keys: Vec<Vec<Key>> = ranks
            .iter()
            .map(|rank| rank_keys(rank, seed, trajectory, steps))
            .collect();
        let mut restore = Restore {
            worker: worker.clone(),
            keys: keys.clone(),
            steps,
            stage: RestoreStage::Pinning(vec![None; ranks.len()]),
        };
        for (index, (rank, keys)) in ranks.iter().zip(keys).enumerate() {
            let pin = Request::Pin {
                pool: rank.pool,
                reader: rank.scope.clone(),
                keys,
            };
            if !self.submit(
                rank.client,
                &pin,
                Purpose::Pin {
                    request,
                    rank: index,
                },
            ) {
                // A server that cannot be asked holds nothing for us; what
                // the others pin is released as they answer.
                restore.stage = RestoreStage::Miss;
                break;
            }
        }
        if restore.stage == RestoreStage::Miss {
            self.orphan_pins(request, &ranks, &restore.keys);
        }
        self.restores.insert(request, restore);
        true
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
        let Some(ranks) = self.workers.get(&restore.worker).cloned() else {
            return;
        };
        match restore.stage {
            RestoreStage::Hit(_) => {
                for (rank, keys) in ranks.iter().zip(restore.keys) {
                    self.unpin(rank, keys);
                }
            }
            RestoreStage::Pinning(answers) => {
                // The ranks that answered release what they pinned now; the
                // others release it when they answer.
                for ((index, rank), answer) in ranks.iter().enumerate().zip(answers) {
                    if let Some(slots) = answer {
                        self.unpin(rank, pinned(&restore.keys[index], &slots));
                    }
                }
                self.orphan_pins(request, &ranks, &restore.keys);
            }
            RestoreStage::Miss => {
                self.orphan_pins(request, &ranks, &restore.keys);
            }
        }
    }

    /// Turn the unanswered pins of `request` into orphans, which release
    /// what they pinned when they are answered.
    fn orphan_pins(&mut self, request: RequestId, ranks: &[RankPool], keys: &[Vec<Key>]) {
        for purpose in self.pending.values_mut() {
            if let Purpose::Pin {
                request: owner,
                rank,
            } = purpose
                && *owner == request
            {
                *purpose = Purpose::Orphan {
                    rank: ranks[*rank].clone(),
                    keys: keys[*rank].clone(),
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
            for purpose in self.pending.values_mut() {
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

    fn unpin(&mut self, rank: &RankPool, keys: Vec<Key>) {
        if keys.is_empty() {
            return;
        }
        let unpin = Request::Unpin {
            pool: rank.pool,
            reader: rank.scope.clone(),
            keys,
        };
        self.submit(rank.client, &unpin, Purpose::Settle);
    }

    /// `rank` answered `request`'s pin with `slots`: once every rank
    /// answered, the restore is a hit when every page is pinned, and a miss
    /// otherwise, which releases the pages pinned.
    fn pinned(&mut self, request: RequestId, rank: usize, slots: Vec<Option<u64>>) {
        let Some(restore) = self.restores.get_mut(&request) else {
            return;
        };
        let RestoreStage::Pinning(answers) = &mut restore.stage else {
            return;
        };
        answers[rank] = Some(slots);
        if !answers.iter().all(Option::is_some) {
            return;
        }
        let answers: Vec<Vec<Option<u64>>> = answers.iter_mut().filter_map(Option::take).collect();
        let Some(ranks) = self.workers.get(&restore.worker).cloned() else {
            return;
        };
        let whole = answers
            .iter()
            .zip(&restore.keys)
            .all(|(slots, keys)| slots.len() == keys.len() && slots.iter().all(Option::is_some));
        if whole {
            let pages = answers
                .iter()
                .zip(&ranks)
                .flat_map(|(slots, rank)| {
                    slots.iter().enumerate().filter_map(|(page, slot)| {
                        slot.map(|slot| LatentPageCopy {
                            rank: rank.scope.rank,
                            pool: rank.pool,
                            slot,
                            page: page as u32,
                        })
                    })
                })
                .collect();
            restore.stage = RestoreStage::Hit(LatentRestore {
                steps: restore.steps,
                pages,
            });
            return;
        }
        restore.stage = RestoreStage::Miss;
        let keys = restore.keys.clone();
        for ((rank, slots), keys) in ranks.iter().zip(answers).zip(keys) {
            self.unpin(rank, pinned(&keys, &slots));
        }
    }

    /// Take the servers' answers. Returns whether a restore or an offload
    /// moved on.
    pub(crate) fn poll(&mut self) -> bool {
        for client in &mut self.clients {
            if let Err(error) = client.pump() {
                tracing::warn!(%error, "a host latent pool connection failed");
            }
        }
        let mut moved = false;
        let ids: Vec<(usize, u64)> = self.pending.keys().copied().collect();
        for id in ids {
            let Some(answer) = self.clients[id.0].take(id.1) else {
                continue;
            };
            let Some(purpose) = self.pending.remove(&id) else {
                continue;
            };
            moved = true;
            match (purpose, answer.reply) {
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
                (Purpose::Pin { request, rank }, reply) => {
                    let slots = match reply {
                        Reply::Pinned { slots } => slots,
                        other => {
                            tracing::warn!(?other, "a host latent pool pin failed");
                            Vec::new()
                        }
                    };
                    self.pinned(request, rank, slots);
                }
                (Purpose::Orphan { rank, keys }, reply) => {
                    if let Reply::Pinned { slots } = reply {
                        self.unpin(&rank, pinned(&keys, &slots));
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
                                pool: rank.pool,
                                keys: reserved,
                            };
                            self.submit(rank.client, &abort, Purpose::Settle);
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
        !self.pending.is_empty()
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

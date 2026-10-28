//! The host KV cache (L2) of the prefix cache, and the KV storage (L3)
//! behind it, as the scheduler drives them.
//!
//! Every host's Storage Server keeps page pools of host memory; each rank
//! of a worker group maps one for each of its cache groups and reports them
//! (`PrefixCacheInfo::page_pools`). The server's Mooncake connector, when it
//! runs one, keeps every committed page in the KV storage and reads pages
//! back into the pools on request. The scheduler decides which pages move,
//! with the reuse rules of each group's attention type ([`super::manager`]),
//! the rules of the device cache (L1):
//!
//! - **Loads.** At admission, after the device prefix match ends at token
//!   `d`, each group's pages from `d` on are pinned in every rank's pool and
//!   looked up, which tells for each page key whether the page is in L2, in
//!   L3 or absent. The hit is the deepest boundary every group's manager
//!   accepts with the pages before `d` on the device and the pages after it
//!   pinned in every rank's pool. When the pages L3 holds would deepen the
//!   hit by at least [`PREFETCH_MIN_TOKENS`] tokens, they are prefetched into
//!   the pools; admission waits for them at most [`prefetch_wait`], then pins
//!   the pages that arrived and takes the hit again. Admission allocates
//!   units for the hit's pages: a full-attention group's pages from `d` to
//!   the boundary, and a sliding-window group's pages of the window before
//!   the boundary that the device does not hold. A sliding window's table
//!   starts at its window, so it maps no earlier position, which no reader
//!   touches: the null block of vLLM. The request's first prefill batch
//!   loads the pages into those units (`Batch::page_loads`) and computes
//!   from the boundary, and the pins end once that call's result is
//!   accepted. Pages that did not arrive in time are computed.
//! - **Offloads.** Once a prompt page is published in L1, a slot is reserved
//!   for it in every rank's pool, while the scheduler holds a reference to
//!   its units. Full attention offloads every published page. A sliding
//!   window offloads only the window before an aligned boundary
//!   ([`KvOffloadManager::offload_boundaries`]): every `boundary_interval`
//!   pages of the largest block size, the last boundary a reissue of the
//!   prompt can reuse, and the last boundary of the published prompt. A page
//!   that falls out of the window before reaching such a window is never
//!   offloaded. The window's pages are reserved as one window (`PageWindow`),
//!   which the pool evicts whole and with the last full-attention page before
//!   its boundary, its anchor; they are reserved while the request's table
//!   still holds them, so the reference keeps their units from being
//!   recycled until the copy commits. The next text batch the worker group
//!   runs copies the pages (`Batch::page_offloads`), and the batch's output
//!   commits them, or aborts them when the batch failed.
//!
//! A page's key chains the keys of its group's pages before it: `h_i =
//! SHA-256(h_{i-1} || group || modality || tokens)` over full pages, each
//! token a little-endian `u32`, from `h_{-1} = SHA-256("uniserve-page" ||
//! content || isolation key)`, where the content is what the ranks report
//! for the group and the isolation key keeps tenants' prefixes apart as L1
//! does. The keys of every group form one key space: for each key, the
//! lookups say where its page is.
//!
//! A world's host KV cache holds one token count on every rank
//! ([`HostCacheDemand`]): the world asks for its device KV cache's tokens
//! times `kv_cache_config.host_ratio`, and each host's Storage Server lowers
//! that for the whole world to what its budget holds over the bytes a token
//! takes on all of the world's ranks there (`Request::HostCache`). Every
//! rank then maps a pool per cache group holding that many tokens
//! (`uniserve.host_cache`); a pool the deployment already has for the same
//! pages, which a previous world created, is shared at its size.
//!
//! A rank's pool is reached through its host's Storage Server: on the
//! engine's host through the server's socket, on another host over the TCP
//! address the rank reports for it. Requests to the servers never block
//! the scheduler: they are submitted, and their answers are taken by
//! [`KvOffloadManager::poll`] at every step; while answers are awaited, a
//! parked scheduler wakes after a millisecond to take them.

use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ipc::{PageCopy, PrefixCacheInfo};
use ipc::{RequestId, UnitId};
use sha2::{Digest, Sha256};
use storage::protocol::{Event, PageAnchor, PageLevel, PageSlot, PageWindow, Reply, Request};
use storage::{Client, Key, Role, Scope};

use super::manager::{self, AttentionManager};
use super::{CachedTokens, GroupShape, PageRef};
use crate::executor::WorkerGroupId;

/// The fewest tokens an L3 hit must add to the pools' hit to be prefetched:
/// a shorter hit is computed faster than it is fetched.
pub(crate) const PREFETCH_MIN_TOKENS: usize = 256;

/// The pages of the largest block size between the aligned boundaries a
/// long prompt offloads its sliding windows before, unless the deployment
/// sets another (`kv_cache_config.boundary_interval`). A sliding window of a
/// thousand tokens in pages of 64 tokens is a quarter of this interval, so
/// the windows offloaded hold at most a quarter of the group's pages.
pub(crate) const DEFAULT_BOUNDARY_INTERVAL: usize = 64;

/// The host KV cache's token count over the device KV cache's when
/// `kv_cache_config.host_ratio` is unset: SGLang's default for a host cache
/// that keeps pages (`--hicache-ratio`).
const DEFAULT_HOST_RATIO: f64 = 2.0;

/// How large a world's host KV cache is (`kv_cache_config.host_ratio` and
/// `host_gib`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HostCacheSize {
    /// The token count over the device KV cache's.
    pub ratio: f64,
    /// The bytes of the deployment's host KV cache on each host, in place
    /// of the budget from the host's available memory.
    pub host_bytes: Option<u64>,
}

impl Default for HostCacheSize {
    fn default() -> Self {
        Self {
            ratio: DEFAULT_HOST_RATIO,
            host_bytes: None,
        }
    }
}

impl HostCacheSize {
    /// The size a deployment states, the defaults where it states none.
    ///
    /// # Errors
    ///
    /// Fails for a ratio that is not a positive finite number.
    pub fn new(ratio: Option<f64>, host_bytes: Option<u64>) -> anyhow::Result<Self> {
        let ratio = ratio.unwrap_or(DEFAULT_HOST_RATIO);
        anyhow::ensure!(
            ratio.is_finite() && ratio > 0.0,
            "kv_cache_config.host_ratio must be a positive number, not {ratio}"
        );
        Ok(Self { ratio, host_bytes })
    }
}

/// What a world's host KV cache asks of each host: the token count of every
/// rank, and the ranks of each host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HostCacheDemand {
    /// The device KV cache's tokens times the ratio, the least over the
    /// world's ranks.
    pub tokens: u64,
    /// By host: where an engine on another host reaches its Storage Server,
    /// and the bytes a token takes on each of the world's ranks there.
    pub hosts: BTreeMap<String, (Option<String>, Vec<u64>)>,
}

impl HostCacheDemand {
    /// The demand of the ranks of `ranks` (by worker group, each rank's host
    /// and cache) whose pages the host KV cache copies, at `ratio`; `None`
    /// when none does. A rank's device KV cache holds its allocatable units,
    /// those beside the sentinel and the units its warmup lends, of the
    /// tokens of the largest page of the groups that hold its context. The
    /// draft cache's group, whose page may span several units' worth of
    /// another group's tokens, holds derived state of those same tokens and
    /// adds none.
    pub(crate) fn of(
        ranks: &BTreeMap<WorkerGroupId, Vec<(String, Option<PrefixCacheInfo>)>>,
        ratio: f64,
    ) -> Option<Self> {
        let mut tokens: Option<u64> = None;
        let mut hosts: BTreeMap<String, (Option<String>, Vec<u64>)> = BTreeMap::new();
        for (host, cache) in ranks.values().flatten() {
            let Some(cache) = cache.as_ref().filter(|cache| cache.host_token_bytes > 0) else {
                continue;
            };
            let unit_tokens = cache
                .groups
                .iter()
                .filter(|group| !group.kind.is_draft())
                .map(|group| u64::from(group.block_size))
                .max()
                .unwrap_or(0);
            let units = cache.num_units.saturating_sub(1 + cache.scratch_units);
            let device = u64::from(units) * unit_tokens;
            let asked = (device as f64 * ratio).ceil() as u64;
            tokens = Some(tokens.map_or(asked, |tokens| tokens.min(asked)));
            let entry = hosts
                .entry(host.clone())
                .or_insert_with(|| (cache.storage_endpoint.clone(), Vec::new()));
            entry.1.push(cache.host_token_bytes);
        }
        Some(Self {
            tokens: tokens?,
            hosts,
        })
    }
}

/// How long admission waits for a prefetch of `tokens` tokens: one second,
/// and a quarter second per 1024 tokens.
pub(crate) fn prefetch_wait(tokens: usize) -> Duration {
    Duration::from_secs(1) + Duration::from_millis(250 * tokens as u64 / 1024)
}

/// One rank's page pool of one cache group.
#[derive(Clone)]
struct RankPool {
    /// The connection to the rank's host's server, in
    /// `KvOffloadManager::clients`.
    client: usize,
    scope: Scope,
    pool: u64,
}

/// The page pools of one cache group of a worker group, one per rank.
#[derive(Clone)]
struct CachePools {
    ranks: Vec<RankPool>,
    content: [u8; 32],
    manager: AttentionManager,
    /// The group holds the draft cache: its pages bound no reuse, a hit
    /// loads its leading pages every rank holds, and a page's key covers
    /// the token after it.
    draft: bool,
}

/// The cache groups of one worker group.
#[derive(Clone)]
struct WorkerPools {
    groups: Vec<CachePools>,
    /// The largest block size of the groups a request needs: every reuse
    /// boundary is a multiple of it.
    step: usize,
}

impl WorkerPools {
    fn managers(&self) -> Vec<AttentionManager> {
        self.groups.iter().map(|group| group.manager).collect()
    }

    /// The reuse rules of the groups a request needs, which bound a reuse.
    fn required_managers(&self) -> Vec<AttentionManager> {
        self.groups
            .iter()
            .filter(|group| !group.draft)
            .map(|group| group.manager)
            .collect()
    }

    fn ranks(&self) -> usize {
        self.groups.first().map_or(0, |group| group.ranks.len())
    }
}

/// What a submitted request was for.
enum Purpose {
    /// Admission pins of `request`'s pages of `group` in `rank`'s pool.
    Pin {
        request: RequestId,
        group: usize,
        rank: usize,
    },
    /// The lookup of `request`'s pages of `group` in `rank`'s pool.
    Lookup {
        request: RequestId,
        group: usize,
        rank: usize,
    },
    /// Pins of `request`'s prefetched pages of `group` in `rank`'s pool, at
    /// these positions of the group's pinned keys.
    Repin {
        request: RequestId,
        group: usize,
        rank: usize,
        positions: Vec<usize>,
    },
    /// Pins of `keys` in `pool` for a request that no longer wants them:
    /// the pages they pinned are unpinned when the answer arrives.
    Orphan { pool: RankPool, keys: Vec<Key> },
    /// The reservation of offload `offload` in `rank`'s pool.
    Reserve { offload: u64, rank: usize },
    /// A request whose answer changes nothing here: commit, abort, unpin,
    /// prefetch.
    Settle,
}

/// Where a request's admission stands.
#[derive(Clone, Copy)]
enum Stage {
    /// Pins and lookups are in flight.
    Pinning,
    /// Pages are being prefetched; `arrived` of `expected` prefetches have
    /// reported.
    Fetching {
        deadline: Instant,
        arrived: usize,
        expected: usize,
    },
    /// The prefetched pages are being pinned; `awaiting` pins have not
    /// answered.
    Repinning { awaiting: usize },
    /// The hit is known.
    Settled,
}

/// One cache group's admission pins of one request.
struct GroupPins {
    /// The first page pinned: every earlier page a reuse needs is on the
    /// device.
    first: usize,
    /// The keys of the group's pages from `first` on.
    keys: Vec<Key>,
    /// Each rank's slots of the keys, once its pin is answered.
    slots: Vec<Option<Vec<Option<u64>>>>,
    /// Each rank's levels of the keys, once its lookup is answered; empty
    /// for keys not looked up.
    levels: Vec<Option<Vec<PageLevel>>>,
    /// The positions of the keys being prefetched, pinned again once they
    /// arrive.
    fetching: Vec<usize>,
}

impl GroupPins {
    /// Whether every rank holds the page at `position` of the keys: pinned
    /// in its pool, or, when `stored` counts, in the pool or the KV storage
    /// at the lookup.
    fn present(&self, position: usize, stored: bool) -> bool {
        self.slots.iter().zip(&self.levels).all(|(slots, levels)| {
            let pinned = slots
                .as_ref()
                .and_then(|slots| slots.get(position))
                .is_some_and(Option::is_some);
            let found = levels
                .as_ref()
                .and_then(|levels| levels.get(position))
                .is_some_and(|level| *level != PageLevel::Absent);
            pinned || (stored && found)
        })
    }

    fn answered(&self) -> bool {
        self.slots.iter().all(Option::is_some) && self.levels.iter().all(Option::is_some)
    }
}

/// The admission pins of one request.
struct Pins {
    worker: WorkerGroupId,
    /// The device prefix match, in tokens, after which the pins start.
    device: usize,
    /// The deepest boundary the request may reuse, in tokens.
    limit: usize,
    groups: Vec<GroupPins>,
    stage: Stage,
}

/// The pages of one cache group a hit loads.
struct GroupHit {
    /// The first page loaded.
    first: usize,
    keys: Vec<Key>,
    /// Each rank's slot of each page.
    slots: Vec<Vec<u64>>,
}

/// A request's hit of the host KV cache, from its admission until its loads
/// are accepted.
struct Hit {
    worker: WorkerGroupId,
    groups: Vec<GroupHit>,
    /// Whether a batch carries the loads.
    sent: bool,
}

/// What an offload of a group's pages is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OffloadKind {
    /// Pages of their own, as full attention offloads them.
    Pages,
    /// The pages of one sliding window, evicted whole, following the
    /// anchor: the cache group and key of the last full-attention page
    /// before the window's boundary.
    Window { anchor: Option<(usize, Key)> },
}

/// The reservations of one offload of published pages.
struct Reservation {
    worker: WorkerGroupId,
    group: usize,
    keys: Vec<Key>,
    pages: Vec<Arc<PageRef>>,
    window: bool,
    answered: usize,
}

/// One page offload waiting for, or carried by, a batch.
struct Offload {
    /// The connection to the pool's server.
    client: usize,
    copy: PageCopy,
    key: Key,
    /// Holds the page's units until the offload is settled.
    _page: Arc<PageRef>,
}

/// One rank's report for [`KvOffloadManager::add_worker`].
pub(crate) struct RankCache<'a> {
    /// Whether the rank runs on the engine's host.
    pub local: bool,
    pub cache: &'a PrefixCacheInfo,
}

/// The scheduler's side of the host KV cache and the KV storage.
pub(crate) struct KvOffloadManager {
    deployment: String,
    /// The engine's world, which publishes the pages it commits.
    world: u64,
    /// The pages of the largest block size between aligned sliding-window
    /// boundaries.
    boundary_interval: usize,
    /// The connections to the servers: the engine's host's first, then other
    /// hosts' by the TCP address they were reached at.
    clients: Vec<Client>,
    remote: HashMap<String, usize>,
    workers: HashMap<WorkerGroupId, WorkerPools>,
    /// Requests submitted and not yet answered, by connection and request.
    pending: HashMap<(usize, u64), Purpose>,
    pins: HashMap<RequestId, Pins>,
    /// The request each prefetch in flight is for, by tag.
    fetches: HashMap<u64, RequestId>,
    next_tag: u64,
    hits: HashMap<RequestId, Hit>,
    reservations: HashMap<u64, Reservation>,
    next_reservation: u64,
    /// Offloads reserved and waiting for a batch of their worker group.
    queued: HashMap<WorkerGroupId, Vec<Offload>>,
    /// Offloads carried by submitted batches, by batch, with the worker
    /// group that runs it.
    carried: HashMap<u64, (WorkerGroupId, Vec<Offload>)>,
}

impl KvOffloadManager {
    /// Connect to this host's Storage Server, at `socket`, as a head of
    /// `world` of `deployment`, offloading sliding windows every
    /// `boundary_interval` pages of a long prompt.
    pub(crate) fn connect(
        socket: &std::path::Path,
        deployment: &str,
        world: u64,
        boundary_interval: usize,
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
            boundary_interval: boundary_interval.max(1),
            clients: vec![client],
            remote: HashMap::new(),
            workers: HashMap::new(),
            pending: HashMap::new(),
            pins: HashMap::new(),
            fetches: HashMap::new(),
            next_tag: 1,
            hits: HashMap::new(),
            reservations: HashMap::new(),
            next_reservation: 1,
            queued: HashMap::new(),
            carried: HashMap::new(),
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
                tracing::warn!(address, %error, "the host KV cache cannot reach a storage server");
                None
            }
        }
    }

    /// The token count every rank of the world holds in the host KV cache
    /// of `host`'s Storage Server, for the world's ranks there that take
    /// `token_bytes` bytes a token (`Request::HostCache`): at most `tokens`.
    /// `local` names the engine's host; another host's server is reached at
    /// `endpoint`.
    ///
    /// # Errors
    ///
    /// Fails when the server cannot be reached or refuses the request.
    pub(crate) fn host_cache_tokens(
        &mut self,
        local: bool,
        endpoint: Option<&str>,
        tokens: u64,
        token_bytes: Vec<u64>,
        host_bytes: Option<u64>,
    ) -> anyhow::Result<u64> {
        let client = if local {
            0
        } else {
            endpoint
                .and_then(|address| self.remote(address))
                .ok_or_else(|| anyhow::anyhow!("another host's Storage Server is unreachable"))?
        };
        let request = Request::HostCache {
            tokens,
            token_bytes,
            host_bytes,
        };
        match self.clients[client].call(&request, &[])?.reply {
            Reply::HostCache { tokens } => Ok(tokens),
            other => {
                anyhow::bail!("the Storage Server answered a host KV cache size with {other:?}")
            }
        }
    }

    /// Serve `worker`, whose ranks (`ranks`, in rank order) are the scopes
    /// of the deployment named by the group and their ranks, when every rank
    /// reports a page pool for every one of its cache groups, of the same
    /// content for a group on every rank, and its server is reachable.
    /// Returns whether the host KV cache serves it.
    pub(crate) fn add_worker(&mut self, worker: &WorkerGroupId, ranks: &[RankCache<'_>]) -> bool {
        let Some(first) = ranks.first() else {
            return false;
        };
        let shapes = &first.cache.groups;
        if shapes.is_empty() {
            return false;
        }
        let mut groups: Vec<CachePools> = shapes
            .iter()
            .map(|shape| CachePools {
                ranks: Vec::with_capacity(ranks.len()),
                content: [0; 32],
                manager: GroupShape::from_group(shape).manager(),
                draft: shape.kind.is_draft(),
            })
            .collect();
        for (rank, report) in ranks.iter().enumerate() {
            if report.cache.groups != *shapes || report.cache.page_pools.len() != shapes.len() {
                return false;
            }
            for pool in &report.cache.page_pools {
                let Some(group) = groups.get_mut(pool.group_id as usize) else {
                    return false;
                };
                if group.ranks.len() != rank || (rank > 0 && group.content != pool.content) {
                    return false;
                }
                group.content = pool.content;
                let client = if report.local {
                    0
                } else {
                    let Some(client) = pool
                        .endpoint
                        .as_deref()
                        .and_then(|address| self.remote(address))
                    else {
                        return false;
                    };
                    client
                };
                group.ranks.push(RankPool {
                    client,
                    scope: Scope {
                        deployment: self.deployment.clone(),
                        group: worker.0.clone(),
                        rank: rank as u32,
                    },
                    pool: pool.pool,
                });
            }
        }
        let step = groups
            .iter()
            .filter(|group| !group.draft)
            .map(|group| group.manager.block_size())
            .max()
            .unwrap_or(1);
        self.workers
            .insert(worker.clone(), WorkerPools { groups, step });
        tracing::info!(worker = %worker.0, groups = shapes.len(), "the host KV cache serves the worker group");
        true
    }

    /// Key `worker`'s pages from now on by `contents`, the seed of each of
    /// its cache groups' pages after its weights changed: the pages written
    /// before keep the old keys, which no later prompt finds. Does nothing
    /// for a worker the host KV cache does not serve.
    pub(crate) fn reseed(
        &mut self,
        worker: &WorkerGroupId,
        contents: &[[u8; 32]],
    ) -> Result<(), String> {
        let Some(pools) = self.workers.get_mut(worker) else {
            return Ok(());
        };
        if pools.groups.len() != contents.len() {
            return Err(format!(
                "worker group {} reports {} page key seeds for {} cache groups",
                worker.0,
                contents.len(),
                pools.groups.len()
            ));
        }
        for (group, content) in pools.groups.iter_mut().zip(contents) {
            group.content = *content;
        }
        Ok(())
    }

    /// Whether the host KV cache serves `worker`.
    pub(crate) fn serves(&self, worker: &WorkerGroupId) -> bool {
        self.workers.contains_key(worker)
    }

    /// The largest block size of `worker`'s cache groups, in tokens: every
    /// reuse boundary is a multiple of it.
    pub(crate) fn step(&self, worker: &WorkerGroupId) -> Option<usize> {
        self.workers.get(worker).map(|pools| pools.step)
    }

    /// The reuse rules of `worker`'s cache groups, in group order.
    pub(crate) fn managers(&self, worker: &WorkerGroupId) -> Vec<AttentionManager> {
        self.workers
            .get(worker)
            .map(WorkerPools::managers)
            .unwrap_or_default()
    }

    /// The first full-attention cache group of `worker`, whose pages anchor
    /// the sliding windows.
    pub(crate) fn anchor_group(&self, worker: &WorkerGroupId) -> Option<usize> {
        self.workers
            .get(worker)?
            .groups
            .iter()
            .position(|group| !group.draft && matches!(group.manager, AttentionManager::Full(_)))
    }

    /// The aligned boundaries in `(from, to]` of a prompt of `prompt_len`
    /// tokens on `worker`, before which its sliding windows are offloaded:
    /// every multiple of `boundary_interval` pages of the largest block
    /// size, the last boundary a reissue of the prompt can reuse (one prompt
    /// token is always computed), and the last boundary of the prompt's
    /// published pages.
    pub(crate) fn offload_boundaries(
        &self,
        worker: &WorkerGroupId,
        prompt_len: usize,
        from: usize,
        to: usize,
    ) -> Vec<usize> {
        let Some(step) = self.step(worker) else {
            return Vec::new();
        };
        let interval = self.boundary_interval * step;
        let reissue = prompt_len.saturating_sub(1) / step * step;
        let published = prompt_len / step * step;
        (from / step + 1..=to / step)
            .map(|page| page * step)
            .filter(|&boundary| {
                boundary % interval == 0 || boundary == reissue || boundary == published
            })
            .collect()
    }

    /// The keys of `worker`'s cache groups for the full pages of the first
    /// `tokens` tokens of `prompt`, by group.
    pub(crate) fn page_keys(
        &self,
        worker: &WorkerGroupId,
        prompt: &[u32],
        isolation_key: Option<u64>,
        tokens: usize,
    ) -> Vec<Vec<Key>> {
        let Some(pools) = self.workers.get(worker) else {
            return Vec::new();
        };
        let mut keys = vec![Vec::new(); pools.groups.len()];
        self.extend_page_keys(
            worker,
            &mut keys,
            CachedTokens::prompt(&prompt[..tokens.min(prompt.len())]),
            isolation_key,
        );
        keys
    }

    /// Extends each group's key chain in `keys` over the full pages of
    /// `tokens` it does not cover yet, as [`Self::page_keys`] chains a
    /// prompt's pages, so that the pages a request generated carry the keys
    /// of the same pages of its migration's prompt. `keys` holds one chain
    /// per group of `worker`, each a prefix of the chain over `tokens`; it is
    /// left as it is for a worker the host KV cache does not serve.
    pub(crate) fn extend_page_keys(
        &self,
        worker: &WorkerGroupId,
        keys: &mut [Vec<Key>],
        tokens: CachedTokens<'_>,
        isolation_key: Option<u64>,
    ) {
        let Some(pools) = self.workers.get(worker) else {
            return;
        };
        for (index, (group, chain)) in pools.groups.iter().zip(keys.iter_mut()).enumerate() {
            let block = group.manager.block_size();
            // A draft cache page's key covers the token after it.
            let key = block + usize::from(group.draft);
            let mut parent: [u8; 32] = match chain.last() {
                Some(last) => last.0,
                None => {
                    let mut digest = Sha256::new();
                    digest.update(b"uniserve-page");
                    digest.update(group.content);
                    match isolation_key {
                        Some(key) => {
                            digest.update([1]);
                            digest.update(key.to_le_bytes());
                        }
                        None => digest.update([0]),
                    }
                    digest.finalize().into()
                }
            };
            for page in chain.len()..(tokens.len() + block).saturating_sub(key) / block {
                let mut digest = Sha256::new();
                digest.update(parent);
                // The group, and text (`TEXT_MODALITY_TAG`).
                digest.update((index as u32).to_le_bytes());
                digest.update([0_u8]);
                for token in tokens.span(page * block..page * block + key).iter() {
                    digest.update(token.to_le_bytes());
                }
                parent = digest.finalize().into();
                chain.push(Key(parent));
            }
        }
    }

    fn submit(&mut self, client: usize, request: &Request, purpose: Purpose) -> bool {
        match self.clients[client].submit(request, &[]) {
            Ok(id) => {
                self.pending.insert((client, id), purpose);
                true
            }
            Err(error) => {
                tracing::warn!(%error, "a host KV cache request failed");
                false
            }
        }
    }

    /// The host KV cache's answer for admitting `request` to `worker`, whose
    /// device cache matched its first `device` tokens and which may reuse at
    /// most `limit`: `None` while pins, lookups or prefetches are in flight,
    /// else the boundary of the hit (at least `device`), which the caller
    /// takes with [`KvOffloadManager::admit`]. `keys` are the request's page
    /// keys by group ([`KvOffloadManager::page_keys`]), covering `limit`
    /// tokens. The first call pins every group's keys from `device` on in
    /// every rank's pool and looks them up. Pins taken for another device
    /// match are released and taken again.
    pub(crate) fn admission(
        &mut self,
        request: RequestId,
        worker: &WorkerGroupId,
        keys: &[Vec<Key>],
        device: usize,
        limit: usize,
    ) -> Option<usize> {
        if let Some(pins) = self.pins.get(&request) {
            if pins.device != device || pins.worker != *worker || pins.limit != limit {
                self.release(request);
            } else {
                return self.advance(request);
            }
        }
        let Some(pools) = self.workers.get(worker).cloned() else {
            return Some(device);
        };
        if limit <= device || keys.len() != pools.groups.len() {
            return Some(device);
        }
        // L3 can only matter when the pages could reach a prefetch.
        let look_up = limit - device >= PREFETCH_MIN_TOKENS;
        let ranks = pools.ranks();
        let mut groups = Vec::with_capacity(pools.groups.len());
        for (index, (group, keys)) in pools.groups.iter().zip(keys).enumerate() {
            let block_size = group.manager.block_size();
            let first = device / block_size;
            let wanted: Vec<Key> = keys
                .get(first..(limit / block_size).min(keys.len()))
                .unwrap_or_default()
                .to_vec();
            let mut pins = GroupPins {
                first,
                keys: wanted.clone(),
                slots: vec![None; ranks],
                levels: vec![if look_up { None } else { Some(Vec::new()) }; ranks],
                fetching: Vec::new(),
            };
            for (rank, pool) in group.ranks.iter().enumerate() {
                if wanted.is_empty() {
                    pins.slots[rank] = Some(Vec::new());
                    pins.levels[rank] = Some(Vec::new());
                    continue;
                }
                let pinned = self.submit(
                    pool.client,
                    &Request::Pin {
                        pool: pool.pool,
                        reader: pool.scope.clone(),
                        keys: wanted.clone(),
                    },
                    Purpose::Pin {
                        request,
                        group: index,
                        rank,
                    },
                );
                if !pinned {
                    // A rank without pins holds nothing of the group.
                    pins.slots[rank] = Some(Vec::new());
                }
                if look_up {
                    let looked = self.submit(
                        pool.client,
                        &Request::Lookup {
                            pool: pool.pool,
                            keys: wanted.clone(),
                        },
                        Purpose::Lookup {
                            request,
                            group: index,
                            rank,
                        },
                    );
                    if !looked {
                        pins.levels[rank] = Some(Vec::new());
                    }
                }
            }
            groups.push(pins);
        }
        self.pins.insert(
            request,
            Pins {
                worker: worker.clone(),
                device,
                limit,
                groups,
                stage: Stage::Pinning,
            },
        );
        self.advance(request)
    }

    /// The deepest boundary of `pins` every manager of `pools` accepts, with
    /// the pages each rank pinned, and, when `stored` counts, the pages each
    /// rank's lookup found in its pool or the KV storage.
    fn boundary(pools: &WorkerPools, pins: &Pins, stored: bool) -> usize {
        let misses: Vec<Vec<usize>> = pools
            .groups
            .iter()
            .zip(&pins.groups)
            .filter(|(group, _)| !group.draft)
            .map(|(group, pins_of)| {
                let pages = pins.limit / group.manager.block_size();
                manager::misses((0..pages).map(|page| {
                    page < pins_of.first || pins_of.present(page - pins_of.first, stored)
                }))
            })
            .collect();
        manager::deepest_boundary(&pools.required_managers(), pools.step, pins.limit, &misses)
            .max(pins.device)
    }

    /// Move `request`'s admission on from its current stage; see
    /// [`KvOffloadManager::admission`].
    fn advance(&mut self, request: RequestId) -> Option<usize> {
        let pins = self.pins.get(&request)?;
        let pools = self.workers.get(&pins.worker)?.clone();
        match pins.stage {
            Stage::Settled => Some(Self::boundary(&pools, pins, false)),
            Stage::Repinning { awaiting } => {
                if awaiting > 0 {
                    return None;
                }
                let pins = self.pins.get_mut(&request)?;
                pins.stage = Stage::Settled;
                Some(Self::boundary(&pools, pins, false))
            }
            Stage::Pinning => {
                if !pins.groups.iter().all(GroupPins::answered) {
                    return None;
                }
                let pinned = Self::boundary(&pools, pins, false);
                let stored = Self::boundary(&pools, pins, true);
                if stored <= pinned || stored - pinned < PREFETCH_MIN_TOKENS {
                    let pins = self.pins.get_mut(&request)?;
                    pins.stage = Stage::Settled;
                    return Some(pinned);
                }
                // Every rank's pool fetches the pages the deeper boundary
                // needs that it did not pin; pages a pool holds already are
                // left as they are.
                let positions: Vec<Vec<usize>> = pools
                    .groups
                    .iter()
                    .zip(&pins.groups)
                    .map(|(group, pins_of)| {
                        let needed = group.manager.needed_pages(stored);
                        (needed.start.max(pins_of.first)..needed.end.max(pins_of.first))
                            .map(|page| page - pins_of.first)
                            .collect()
                    })
                    .collect();
                let mut fetches = Vec::new();
                for ((group, pins_of), positions) in
                    pools.groups.iter().zip(&pins.groups).zip(&positions)
                {
                    for (rank, pool) in group.ranks.iter().enumerate() {
                        let slots = pins_of.slots[rank].as_deref().unwrap_or_default();
                        let keys: Vec<Key> = positions
                            .iter()
                            .filter(|&&position| {
                                slots.get(position).is_none_or(|slot| slot.is_none())
                            })
                            .map(|&position| pins_of.keys[position])
                            .collect();
                        if !keys.is_empty() {
                            fetches.push((pool.clone(), keys));
                        }
                    }
                }
                let mut expected = 0;
                for (pool, keys) in fetches {
                    let tag = self.next_tag;
                    self.next_tag += 1;
                    if self.submit(
                        pool.client,
                        &Request::Prefetch {
                            pool: pool.pool,
                            keys,
                            tag,
                        },
                        Purpose::Settle,
                    ) {
                        self.fetches.insert(tag, request);
                        expected += 1;
                    }
                }
                let pins = self.pins.get_mut(&request)?;
                if expected == 0 {
                    pins.stage = Stage::Settled;
                    return Some(pinned);
                }
                for (pins_of, positions) in pins.groups.iter_mut().zip(positions) {
                    pins_of.fetching = positions;
                }
                pins.stage = Stage::Fetching {
                    deadline: Instant::now() + prefetch_wait(stored - pinned),
                    arrived: 0,
                    expected,
                };
                None
            }
            Stage::Fetching {
                deadline,
                arrived,
                expected,
            } => {
                if arrived < expected && Instant::now() < deadline {
                    return None;
                }
                // Pin what arrived; pages still in flight are computed, and
                // stay in the pools for later requests.
                self.fetches.retain(|_, waiting| *waiting != request);
                let mut repins = Vec::new();
                for (index, (group, pins_of)) in pools.groups.iter().zip(&pins.groups).enumerate() {
                    for (rank, pool) in group.ranks.iter().enumerate() {
                        let Some(slots) = pins_of.slots[rank].as_ref() else {
                            continue;
                        };
                        let positions: Vec<usize> = pins_of
                            .fetching
                            .iter()
                            .copied()
                            .filter(|&position| slots.get(position).is_some_and(Option::is_none))
                            .collect();
                        if !positions.is_empty() {
                            let keys = positions
                                .iter()
                                .map(|&position| pins_of.keys[position])
                                .collect();
                            repins.push((index, rank, pool.clone(), keys, positions));
                        }
                    }
                }
                let mut awaiting = 0;
                for (group, rank, pool, keys, positions) in repins {
                    if self.submit(
                        pool.client,
                        &Request::Pin {
                            pool: pool.pool,
                            reader: pool.scope.clone(),
                            keys,
                        },
                        Purpose::Repin {
                            request,
                            group,
                            rank,
                            positions,
                        },
                    ) {
                        awaiting += 1;
                    }
                }
                let pins = self.pins.get_mut(&request)?;
                if awaiting == 0 {
                    pins.stage = Stage::Settled;
                    return Some(Self::boundary(&pools, pins, false));
                }
                pins.stage = Stage::Repinning { awaiting };
                None
            }
        }
    }

    /// Admit `request` with its pinned pages: returns the device match and
    /// the hit's boundary, when the hit goes past the match. The pins of
    /// pages the hit does not load end now; the hit's when its loads are
    /// done ([`KvOffloadManager::loaded`]) or the request leaves
    /// ([`KvOffloadManager::sweep`]).
    pub(crate) fn admit(&mut self, request: RequestId) -> Option<(usize, usize)> {
        let pins = self.pins.remove(&request)?;
        self.fetches.retain(|_, waiting| *waiting != request);
        let pools = self.workers.get(&pins.worker)?.clone();
        let boundary = Self::boundary(&pools, &pins, false);
        let mut groups = Vec::with_capacity(pools.groups.len());
        for (group, pins_of) in pools.groups.iter().zip(&pins.groups) {
            // A full-attention group loads its pages from the device match
            // to the boundary; a sliding window those of its window; the
            // draft cache its leading pages every rank pinned.
            let needed = group.manager.needed_pages(boundary);
            let mut loaded = needed.start.max(pins_of.first)..needed.end.max(pins_of.first);
            if group.draft {
                let held = (0..loaded.end - loaded.start)
                    .take_while(|&offset| {
                        pins_of.present(loaded.start - pins_of.first + offset, false)
                    })
                    .count();
                loaded.end = loaded.start + held;
            }
            let positions = loaded.start - pins_of.first..loaded.end - pins_of.first;
            let mut slots = Vec::with_capacity(group.ranks.len());
            for (rank, pool) in group.ranks.iter().enumerate() {
                let pinned = pins_of.slots[rank].clone().unwrap_or_default();
                let extra: Vec<Key> = pins_of
                    .keys
                    .iter()
                    .zip(&pinned)
                    .enumerate()
                    .filter(|(position, (_, slot))| slot.is_some() && !positions.contains(position))
                    .map(|(_, (key, _))| *key)
                    .collect();
                if !extra.is_empty() {
                    self.unpin(pool, extra);
                }
                slots.push(
                    positions
                        .clone()
                        .filter_map(|position| pinned.get(position).copied().flatten())
                        .collect(),
                );
            }
            groups.push(GroupHit {
                first: loaded.start,
                keys: pins_of.keys[positions].to_vec(),
                slots,
            });
        }
        if boundary <= pins.device {
            return None;
        }
        self.hits.insert(
            request,
            Hit {
                worker: pins.worker,
                groups,
                sent: false,
            },
        );
        Some((pins.device, boundary))
    }

    /// Release `request`'s admission pins: it left the queue, or its device
    /// match changed.
    pub(crate) fn release(&mut self, request: RequestId) {
        let Some(pins) = self.pins.remove(&request) else {
            return;
        };
        self.fetches.retain(|_, waiting| *waiting != request);
        let Some(pools) = self.workers.get(&pins.worker).cloned() else {
            return;
        };
        // Pins still in flight are released when their answers arrive, for
        // exactly the pages they pinned: a pool counts pins per reader, so an
        // unpin of a page this request never pinned would end another
        // request's pin.
        for purpose in self.pending.values_mut() {
            let orphan = match purpose {
                Purpose::Pin {
                    request: pinned,
                    group,
                    rank,
                } if *pinned == request => Some((*group, *rank, pins.groups[*group].keys.clone())),
                Purpose::Repin {
                    request: pinned,
                    group,
                    rank,
                    positions,
                } if *pinned == request => Some((
                    *group,
                    *rank,
                    positions
                        .iter()
                        .map(|position| pins.groups[*group].keys[*position])
                        .collect(),
                )),
                Purpose::Lookup {
                    request: looked, ..
                } if *looked == request => {
                    *purpose = Purpose::Settle;
                    None
                }
                _ => None,
            };
            if let Some((group, rank, keys)) = orphan
                && let Some(pool) = pools
                    .groups
                    .get(group)
                    .and_then(|group| group.ranks.get(rank))
            {
                *purpose = Purpose::Orphan {
                    pool: pool.clone(),
                    keys,
                };
            }
        }
        for (group, pins_of) in pools.groups.iter().zip(&pins.groups) {
            for (rank, pool) in group.ranks.iter().enumerate() {
                let pinned = Self::pinned(&pins_of.keys, pins_of.slots[rank].as_deref());
                if !pinned.is_empty() {
                    self.unpin(pool, pinned);
                }
            }
        }
    }

    /// The keys whose pages `slots`, a pin's answer for `keys`, pinned.
    fn pinned(keys: &[Key], slots: Option<&[Option<u64>]>) -> Vec<Key> {
        keys.iter()
            .zip(slots.unwrap_or_default())
            .filter(|(_, slot)| slot.is_some())
            .map(|(key, _)| *key)
            .collect()
    }

    fn unpin(&mut self, pool: &RankPool, keys: Vec<Key>) {
        self.submit(
            pool.client,
            &Request::Unpin {
                pool: pool.pool,
                reader: pool.scope.clone(),
                keys,
            },
            Purpose::Settle,
        );
    }

    /// The page loads of `request`'s hit, once: `units(group, page)` names
    /// the units of an absolute page of the request's table of `group`,
    /// freshly allocated for the hit. Empty for a request without a hit or
    /// whose loads a batch already carries.
    pub(crate) fn loads(
        &mut self,
        request: RequestId,
        units: impl Fn(usize, usize) -> Option<Vec<UnitId>>,
    ) -> Vec<PageCopy> {
        let Some(hit) = self.hits.get_mut(&request).filter(|hit| !hit.sent) else {
            return Vec::new();
        };
        let Some(pools) = self.workers.get(&hit.worker) else {
            return Vec::new();
        };
        hit.sent = true;
        let mut copies = Vec::new();
        for (index, (group, hit_of)) in pools.groups.iter().zip(&hit.groups).enumerate() {
            for (rank, pool) in group.ranks.iter().enumerate() {
                for (offset, slot) in hit_of.slots[rank].iter().enumerate() {
                    if let Some(units) = units(index, hit_of.first + offset) {
                        copies.push(PageCopy {
                            rank: rank as u32,
                            pool: pool.pool,
                            group_id: index as u32,
                            slot: *slot,
                            units,
                        });
                    }
                }
            }
        }
        copies
    }

    /// The pages of `request`'s hit of each cache group, which hold the
    /// host KV cache's bytes already; empty without a hit.
    pub(crate) fn hit_pages(&self, request: RequestId) -> Vec<Range<usize>> {
        self.hits.get(&request).map_or_else(Vec::new, |hit| {
            hit.groups
                .iter()
                .map(|group| group.first..group.first + group.keys.len())
                .collect()
        })
    }

    /// `request`'s loads were accepted: its pins end.
    pub(crate) fn loaded(&mut self, request: RequestId) {
        if self.hits.get(&request).is_some_and(|hit| hit.sent) {
            self.drop_hit(request);
        }
    }

    fn drop_hit(&mut self, request: RequestId) {
        let Some(hit) = self.hits.remove(&request) else {
            return;
        };
        let Some(pools) = self.workers.get(&hit.worker).cloned() else {
            return;
        };
        for (group, hit_of) in pools.groups.iter().zip(hit.groups) {
            if hit_of.keys.is_empty() {
                continue;
            }
            for pool in &group.ranks {
                self.unpin(pool, hit_of.keys.clone());
            }
        }
    }

    /// End `request`'s pins and hit: it does not use them.
    pub(crate) fn abandon(&mut self, request: RequestId) {
        self.release(request);
        self.drop_hit(request);
    }

    /// End the pins and hits of requests `live` no longer names: they were
    /// cancelled, failed or finished.
    pub(crate) fn sweep(&mut self, live: impl Fn(RequestId) -> bool) {
        let gone: Vec<RequestId> = self.pins.keys().copied().filter(|id| !live(*id)).collect();
        for request in gone {
            self.release(request);
        }
        let gone: Vec<RequestId> = self.hits.keys().copied().filter(|id| !live(*id)).collect();
        for request in gone {
            self.drop_hit(request);
        }
    }

    /// Whether answers are awaited, which the scheduler polls for.
    pub(crate) fn waiting(&self) -> bool {
        !self.pending.is_empty()
            || self
                .pins
                .values()
                .any(|pins| matches!(pins.stage, Stage::Fetching { .. }))
    }

    /// Offload published pages of `worker`'s cache group `group` to the host
    /// KV cache: `keys[i]` is the key of the page `pages[i]` references. A
    /// slot is reserved for each page in every rank's pool; a reserved page
    /// waits for a batch of `worker`. A window that a rank's pool cannot
    /// reserve whole is not offloaded to it.
    pub(crate) fn offload(
        &mut self,
        worker: &WorkerGroupId,
        group: usize,
        keys: Vec<Key>,
        pages: Vec<PageRef>,
        kind: OffloadKind,
    ) {
        let Some(pools) = self.workers.get(worker).cloned() else {
            return;
        };
        let Some(cache) = pools.groups.get(group) else {
            return;
        };
        if keys.is_empty() || keys.len() != pages.len() {
            return;
        }
        let id = self.next_reservation;
        self.next_reservation += 1;
        let mut submitted = 0;
        for (rank, pool) in cache.ranks.iter().enumerate() {
            let window = match kind {
                OffloadKind::Pages => None,
                OffloadKind::Window { anchor } => Some(PageWindow {
                    anchor: anchor.and_then(|(group, key)| {
                        let pool = pools.groups.get(group)?.ranks.get(rank)?.pool;
                        Some(PageAnchor { pool, key })
                    }),
                }),
            };
            if self.submit(
                pool.client,
                &Request::Reserve {
                    pool: pool.pool,
                    writer: pool.scope.clone(),
                    keys: keys.clone(),
                    window,
                },
                Purpose::Reserve { offload: id, rank },
            ) {
                submitted += 1;
            }
        }
        if submitted > 0 {
            self.reservations.insert(
                id,
                Reservation {
                    worker: worker.clone(),
                    group,
                    keys,
                    pages: pages.into_iter().map(Arc::new).collect(),
                    window: matches!(kind, OffloadKind::Window { .. }),
                    answered: cache.ranks.len() - submitted,
                },
            );
        }
    }

    /// Whether published pages are still on their way to the host KV cache:
    /// slots being reserved, offloads waiting for or carried by a batch, or
    /// their commits unanswered.
    pub(crate) fn offloading(&self) -> bool {
        !self.reservations.is_empty()
            || self.queued.values().any(|offloads| !offloads.is_empty())
            || !self.carried.is_empty()
            || self
                .pending
                .values()
                .any(|purpose| matches!(purpose, Purpose::Reserve { .. } | Purpose::Settle))
    }

    /// The worker groups whose reserved page offloads wait for a batch.
    pub(crate) fn queued_workers(&self) -> Vec<WorkerGroupId> {
        self.queued
            .iter()
            .filter(|(_, offloads)| !offloads.is_empty())
            .map(|(worker, _)| worker.clone())
            .collect()
    }

    /// The page offloads queued for `worker`, which the batch `batch`
    /// carries.
    pub(crate) fn take_offloads(&mut self, worker: &WorkerGroupId, batch: u64) -> Vec<PageCopy> {
        let Some(offloads) = self.queued.remove(worker) else {
            return Vec::new();
        };
        let copies: Vec<PageCopy> = offloads
            .iter()
            .map(|offload| offload.copy.clone())
            .collect();
        tracing::debug!(
            worker = %worker.0,
            batch,
            pages = copies.len(),
            "a batch carries page offloads to the host KV cache"
        );
        self.carried
            .entry(batch)
            .or_insert_with(|| (worker.clone(), Vec::new()))
            .1
            .extend(offloads);
        copies
    }

    /// The batch `batch` completed: the pages it offloaded are committed
    /// when every call of it succeeded, and aborted otherwise.
    pub(crate) fn completed(&mut self, batch: u64, succeeded: bool) {
        let Some((_, offloads)) = self.carried.remove(&batch) else {
            return;
        };
        self.settle(offloads, succeeded);
    }

    fn settle(&mut self, offloads: Vec<Offload>, succeeded: bool) {
        let mut by_pool: HashMap<(usize, u64), Vec<Key>> = HashMap::new();
        for offload in &offloads {
            by_pool
                .entry((offload.client, offload.copy.pool))
                .or_default()
                .push(offload.key);
        }
        for ((client, pool), keys) in by_pool {
            let request = if succeeded {
                Request::Commit { pool, keys }
            } else {
                Request::Abort { pool, keys }
            };
            self.submit(client, &request, Purpose::Settle);
        }
    }

    /// `worker`'s ranks were replaced: the offloads waiting for its batches
    /// and those its batches carried are aborted, and its admission pins
    /// released.
    pub(crate) fn worker_failed(&mut self, worker: &WorkerGroupId) {
        let mut lost = self.queued.remove(worker).unwrap_or_default();
        let batches: Vec<u64> = self
            .carried
            .iter()
            .filter(|(_, (carrier, _))| carrier == worker)
            .map(|(batch, _)| *batch)
            .collect();
        for batch in batches {
            if let Some((_, offloads)) = self.carried.remove(&batch) {
                lost.extend(offloads);
            }
        }
        self.settle(lost, false);
        let requests: Vec<RequestId> = self
            .pins
            .iter()
            .filter(|(_, pins)| pins.worker == *worker)
            .map(|(request, _)| *request)
            .collect();
        for request in requests {
            self.release(request);
        }
    }

    /// Take the servers' answers and events. Returns whether an admission
    /// moved on, which may let the waiting queue's head proceed.
    pub(crate) fn poll(&mut self) -> bool {
        let mut admissions = false;
        for client in &mut self.clients {
            if let Err(error) = client.pump() {
                tracing::warn!(%error, "a host KV cache connection failed");
            }
            while let Some((event, _)) = client.next_event() {
                let Event::Prefetched { tag, .. } = event else {
                    continue;
                };
                let Some(request) = self.fetches.remove(&tag) else {
                    continue;
                };
                if let Some(Pins {
                    stage: Stage::Fetching { arrived, .. },
                    ..
                }) = self.pins.get_mut(&request)
                {
                    *arrived += 1;
                    admissions = true;
                }
            }
        }
        // A prefetch past its deadline lets its admission move on.
        admissions |= self.pins.values().any(|pins| {
            matches!(pins.stage, Stage::Fetching { deadline, .. } if Instant::now() >= deadline)
        });
        let ids: Vec<(usize, u64)> = self.pending.keys().copied().collect();
        for id in ids {
            let Some(answer) = self.clients[id.0].take(id.1) else {
                continue;
            };
            let Some(purpose) = self.pending.remove(&id) else {
                continue;
            };
            match (purpose, answer.reply) {
                (
                    Purpose::Pin {
                        request,
                        group,
                        rank,
                    },
                    reply,
                ) => {
                    let slots = match reply {
                        Reply::Pinned { slots } => slots,
                        other => {
                            tracing::warn!(?other, "a host KV cache pin failed");
                            Vec::new()
                        }
                    };
                    if let Some(pins) = self.pins.get_mut(&request) {
                        pins.groups[group].slots[rank] = Some(slots);
                        admissions = true;
                    }
                }
                (
                    Purpose::Lookup {
                        request,
                        group,
                        rank,
                    },
                    reply,
                ) => {
                    let levels = match reply {
                        Reply::Lookup { levels } => levels,
                        other => {
                            tracing::warn!(?other, "a host KV cache lookup failed");
                            Vec::new()
                        }
                    };
                    if let Some(pins) = self.pins.get_mut(&request) {
                        pins.groups[group].levels[rank] = Some(levels);
                        admissions = true;
                    }
                }
                (
                    Purpose::Repin {
                        request,
                        group,
                        rank,
                        positions,
                    },
                    reply,
                ) => {
                    let slots = match reply {
                        Reply::Pinned { slots } => slots,
                        other => {
                            tracing::warn!(?other, "a host KV cache pin failed");
                            Vec::new()
                        }
                    };
                    if let Some(pins) = self.pins.get_mut(&request) {
                        if let Some(pinned) = pins.groups[group].slots[rank].as_mut() {
                            for (position, slot) in positions.into_iter().zip(slots) {
                                if let Some(entry) = pinned.get_mut(position) {
                                    *entry = slot;
                                }
                            }
                        }
                        if let Stage::Repinning { awaiting } = &mut pins.stage {
                            *awaiting = awaiting.saturating_sub(1);
                        }
                        admissions = true;
                    }
                }
                (Purpose::Orphan { pool, keys }, reply) => {
                    if let Reply::Pinned { slots } = reply {
                        let pinned = Self::pinned(&keys, Some(&slots));
                        if !pinned.is_empty() {
                            self.unpin(&pool, pinned);
                        }
                    }
                }
                (Purpose::Reserve { offload, rank }, reply) => {
                    let slots = match reply {
                        Reply::Reserved { slots } => slots,
                        other => {
                            tracing::warn!(?other, "a host KV cache reservation failed");
                            Vec::new()
                        }
                    };
                    self.reserved(offload, rank, &slots);
                }
                (Purpose::Settle, Reply::Error { kind, message }) => {
                    tracing::warn!(?kind, message, "a host KV cache request failed");
                }
                (Purpose::Settle, _) => {}
            }
        }
        admissions
    }

    /// `rank` answered reservation `id` with `slots`: its reserved pages
    /// wait for a batch. A window whose pages the pool could not all store
    /// is aborted on that rank: a window missing one page is useless.
    fn reserved(&mut self, id: u64, rank: usize, slots: &[PageSlot]) {
        let Some(reservation) = self.reservations.get_mut(&id) else {
            return;
        };
        let pool = self
            .workers
            .get(&reservation.worker)
            .and_then(|pools| pools.groups.get(reservation.group))
            .and_then(|group| group.ranks.get(rank))
            .map(|pool| (pool.client, pool.pool));
        let whole = slots.len() == reservation.keys.len()
            && slots.iter().all(|slot| !matches!(slot, PageSlot::Full));
        let mut aborted = None;
        if let Some((client, pool)) = pool {
            if reservation.window && !whole {
                let keys: Vec<Key> = reservation
                    .keys
                    .iter()
                    .zip(slots)
                    .filter(|(_, slot)| matches!(slot, PageSlot::Reserved { .. }))
                    .map(|(key, _)| *key)
                    .collect();
                aborted = Some((client, pool, keys));
            } else {
                for ((key, page), slot) in
                    reservation.keys.iter().zip(&reservation.pages).zip(slots)
                {
                    if let PageSlot::Reserved { slot } = slot {
                        self.queued
                            .entry(reservation.worker.clone())
                            .or_default()
                            .push(Offload {
                                client,
                                copy: PageCopy {
                                    rank: rank as u32,
                                    pool,
                                    group_id: reservation.group as u32,
                                    slot: *slot,
                                    units: page.units().to_vec(),
                                },
                                key: *key,
                                _page: Arc::clone(page),
                            });
                    }
                }
            }
        }
        reservation.answered += 1;
        let done = self
            .workers
            .get(&reservation.worker)
            .and_then(|pools| pools.groups.get(reservation.group))
            .is_none_or(|group| reservation.answered >= group.ranks.len());
        if done {
            self.reservations.remove(&id);
        }
        if let Some((client, pool, keys)) = aborted
            && !keys.is_empty()
        {
            self.submit(client, &Request::Abort { pool, keys }, Purpose::Settle);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::os::fd::{AsFd, AsRawFd, FromRawFd, OwnedFd};
    use std::sync::atomic::{AtomicBool, Ordering};

    use ipc::PagePoolInfo;
    use ipc::{CacheGroup, CacheGroupKind, PrefixCacheDtype};
    use storage::Claim;
    use storage::protocol::PageShape;
    use storage_server::Server;

    use super::*;
    use crate::kv_cache::BlockPool;

    /// A Storage Server on a socket of its own, with pools of 64 pages of
    /// 16 bytes, stopped when dropped.
    struct Running {
        socket: std::path::PathBuf,
        stop: Arc<AtomicBool>,
        wake: OwnedFd,
        thread: Option<std::thread::JoinHandle<()>>,
        _directory: tempfile::TempDir,
    }

    impl Running {
        fn start() -> Self {
            let directory = tempfile::tempdir().unwrap();
            let socket = directory.path().join("storage.sock");
            let (listener, lock) = storage::socket::bind(&socket).unwrap().unwrap();
            let mut ends = [0; 2];
            // SAFETY: pipe2 fills the two descriptors it creates.
            let created = unsafe { libc::pipe2(ends.as_mut_ptr(), libc::O_NONBLOCK) };
            assert_eq!(created, 0);
            // SAFETY: two new descriptors this test owns.
            let (reader, writer) =
                unsafe { (OwnedFd::from_raw_fd(ends[0]), OwnedFd::from_raw_fd(ends[1])) };
            let stop = Arc::new(AtomicBool::new(false));
            let stopped = Arc::clone(&stop);
            let thread = std::thread::spawn(move || {
                let _lock = lock;
                let mut server = Server::new(listener, 64 * 16, 0, None, None).unwrap();
                server
                    .run(reader.as_fd(), &|| stopped.load(Ordering::Acquire))
                    .unwrap();
            });
            Self {
                socket,
                stop,
                wake: writer,
                thread: Some(thread),
                _directory: directory,
            }
        }
    }

    impl Drop for Running {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Release);
            // SAFETY: a write to a pipe this test owns.
            unsafe { libc::write(self.wake.as_raw_fd(), b"x".as_ptr().cast(), 1) };
            if let Some(thread) = self.thread.take() {
                thread.join().unwrap();
            }
        }
    }

    /// A connector standing in for Mooncake, which holds every page and
    /// reads each into its slot, until `stop` reads true.
    fn connector(socket: &std::path::Path, stop: Arc<AtomicBool>) -> std::thread::JoinHandle<()> {
        let mut client = Client::connect(socket, Role::Connector).unwrap();
        std::thread::spawn(move || {
            while !stop.load(Ordering::Acquire) {
                client.pump().unwrap();
                while let Some((event, _)) = client.next_event() {
                    let (task, pages) = match event {
                        Event::Exists { task, keys, .. } => (task, keys.len()),
                        Event::Download { task, pages, .. } | Event::Upload { task, pages, .. } => {
                            (task, pages.len())
                        }
                        _ => continue,
                    };
                    let finished = Request::Finished {
                        task,
                        results: vec![true; pages],
                    };
                    client.submit(&finished, &[]).unwrap();
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        })
    }

    /// A worker of one rank on `server`, which maps a page pool for each of
    /// `contents`, and the pools.
    fn worker_pools(server: &Running, contents: &[u8]) -> (Client, Vec<u64>) {
        let scope = Scope {
            deployment: "model".into(),
            group: "text".into(),
            rank: 0,
        };
        let mut worker = Client::connect(
            &server.socket,
            Role::Worker {
                scope,
                generation: 1,
                claim: Claim([0; 16]),
                world: 0,
            },
        )
        .unwrap();
        let pools = contents
            .iter()
            .map(|content| {
                let shape = PageShape {
                    layout: Key([*content; 32]),
                    kv_head_offset: 0,
                    kv_head_count: 8,
                    page_bytes: 16,
                };
                let request = Request::PagePool {
                    shape,
                    numa_node: None,
                    bytes: None,
                };
                let Reply::PagePool { pool, .. } = worker.call(&request, &[]).unwrap().reply else {
                    panic!("the page pool was refused");
                };
                pool
            })
            .collect();
        (worker, pools)
    }

    /// A cache group of pages of 64 tokens, of full attention or of a
    /// sliding window of `window` tokens.
    fn cache_group(window: Option<u32>) -> CacheGroup {
        CacheGroup {
            kind: window.map_or(CacheGroupKind::Full, |window| {
                CacheGroupKind::SlidingWindow { window }
            }),
            block_size: 64,
            units_per_page: 1,
            layer_ids: vec![0],
            num_kv_heads: 8,
            total_kv_heads: 8,
            kv_head_offset: 0,
            head_dim: 128,
        }
    }

    /// The prefix cache report of a rank with `groups`, whose page pools
    /// are `pools` of contents `contents`.
    fn report(groups: Vec<CacheGroup>, pools: &[u64], contents: &[u8]) -> PrefixCacheInfo {
        PrefixCacheInfo {
            num_units: 64,
            dtype: PrefixCacheDtype::BFloat16,
            groups,
            page_pools: pools
                .iter()
                .zip(contents)
                .enumerate()
                .map(|(group, (pool, content))| PagePoolInfo {
                    group_id: group as u32,
                    pool: *pool,
                    content: [*content; 32],
                    endpoint: None,
                })
                .collect(),
            scratch_units: 0,
            segments: Vec::new(),
            held_segments: 0,
            inherited_start: 0,
            inherited_units: 0,
            identity: String::new(),
            host_token_bytes: 0,
            storage_endpoint: None,
        }
    }

    /// A world asks for its device KV cache's tokens times the ratio, the
    /// least over its ranks, and states the bytes a token takes on each rank
    /// by host; ranks that copy no page ask nothing.
    #[test]
    fn a_world_asks_its_hosts_for_one_token_count() {
        let rank = |host: &str, units: u32, scratch: u32, bytes: u64| {
            let mut cache = report(vec![cache_group(None)], &[], &[]);
            cache.num_units = units;
            cache.scratch_units = scratch;
            cache.host_token_bytes = bytes;
            cache.storage_endpoint = Some(format!("{host}:7000"));
            (host.to_owned(), Some(cache))
        };
        let block = u64::from(cache_group(None).block_size);
        let ranks = BTreeMap::from([
            (
                WorkerGroupId("text".to_owned()),
                vec![rank("a", 65, 0, 100), rank("b", 65, 0, 100)],
            ),
            (
                WorkerGroupId("prefill".to_owned()),
                vec![rank("a", 41, 8, 300), ("b".to_owned(), None)],
            ),
        ]);

        let demand = HostCacheDemand::of(&ranks, 1.5).unwrap();
        // The prefill ranks allocate 32 units beside the sentinel and the
        // units their warmup lends.
        assert_eq!(demand.tokens, (32 * block * 3).div_ceil(2));
        assert_eq!(
            demand.hosts["a"],
            (Some("a:7000".to_owned()), vec![300, 100])
        );
        assert_eq!(demand.hosts["b"], (Some("b:7000".to_owned()), vec![100]));

        let none = BTreeMap::from([(
            WorkerGroupId("experts".to_owned()),
            vec![("a".to_owned(), None)],
        )]);
        assert_eq!(HostCacheDemand::of(&none, 2.0), None);
    }

    /// A rank whose pool also keeps the draft cache, a group whose page
    /// spans sixteen of the context group's pages, asks the host KV cache
    /// for the tokens of its context alone, as the same rank without it.
    #[test]
    fn the_draft_cache_adds_no_tokens_to_the_host_cache_demand() {
        let ranks = |groups: Vec<CacheGroup>| {
            let mut cache = report(groups, &[], &[]);
            cache.host_token_bytes = 100;
            BTreeMap::from([(
                WorkerGroupId("text".to_owned()),
                vec![("a".to_owned(), Some(cache))],
            )])
        };
        let draft = CacheGroup {
            kind: CacheGroupKind::Draft,
            block_size: 16 * cache_group(None).block_size,
            layer_ids: vec![1],
            ..cache_group(None)
        };

        let context = HostCacheDemand::of(&ranks(vec![cache_group(None)]), 2.0).unwrap();
        let drafting = HostCacheDemand::of(&ranks(vec![cache_group(None), draft]), 2.0).unwrap();
        // 63 allocatable units of 64-token pages, twice.
        assert_eq!(context.tokens, 63 * 64 * 2);
        assert_eq!(drafting.tokens, context.tokens);
    }

    /// Poll `kv_offload` until `done` holds, for at most ten seconds.
    fn poll_until(
        kv_offload: &mut KvOffloadManager,
        mut done: impl FnMut(&mut KvOffloadManager) -> bool,
    ) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !done(kv_offload) {
            assert!(
                Instant::now() < deadline,
                "the host KV cache did not answer"
            );
            kv_offload.poll();
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn admission_prefetches_a_long_distributed_hit_and_pins_what_arrived() {
        let server = Running::start();
        let (_worker, pools) = worker_pools(&server, &[3]);
        let stop = Arc::new(AtomicBool::new(false));
        let mooncake = connector(&server.socket, Arc::clone(&stop));

        let mut kv_offload =
            KvOffloadManager::connect(&server.socket, "model", 0, DEFAULT_BOUNDARY_INTERVAL)
                .unwrap();
        let cache = report(vec![cache_group(None)], &pools, &[3]);
        let text = WorkerGroupId("text".into());
        assert!(kv_offload.add_worker(
            &text,
            &[RankCache {
                local: true,
                cache: &cache,
            }]
        ));

        // Eight pages that only L3 holds: 512 tokens are worth a prefetch,
        // and every page arrives within the wait.
        let prompt: Vec<u32> = (0..8 * 64 + 1).collect();
        let keys = kv_offload.page_keys(&text, &prompt, None, 512);
        let mut boundary = None;
        poll_until(&mut kv_offload, |kv_offload| {
            boundary = kv_offload.admission(RequestId(1), &text, &keys, 0, 512);
            boundary.is_some()
        });
        assert_eq!(boundary, Some(512));
        assert_eq!(kv_offload.admit(RequestId(1)), Some((0, 512)));

        stop.store(true, Ordering::Release);
        mooncake.join().unwrap();
    }

    /// A hybrid worker's hit ends at the deepest boundary whose sliding
    /// window the host KV cache holds, though its full-attention pages go
    /// further: the full-attention group loads every page before the
    /// boundary, the sliding window only the pages of its window.
    #[test]
    fn a_hybrid_hit_ends_at_the_deepest_window_the_host_holds() {
        let server = Running::start();
        let (_worker, pools) = worker_pools(&server, &[4, 5]);
        let mut kv_offload =
            KvOffloadManager::connect(&server.socket, "model", 0, DEFAULT_BOUNDARY_INTERVAL)
                .unwrap();
        let cache = report(
            vec![cache_group(None), cache_group(Some(128))],
            &pools,
            &[4, 5],
        );
        let text = WorkerGroupId("text".into());
        assert!(kv_offload.add_worker(
            &text,
            &[RankCache {
                local: true,
                cache: &cache,
            }]
        ));

        // A request offloaded ten full-attention pages and the window before
        // token 384 (pages 4 and 5), anchored to the full page before it.
        let prompt: Vec<u32> = (0..10 * 64 + 1).collect();
        let keys = kv_offload.page_keys(&text, &prompt, None, prompt.len());
        let block_pool = BlockPool::new(64, 0);
        kv_offload.offload(
            &text,
            0,
            keys[0][..10].to_vec(),
            block_pool.allocate(10, 1).unwrap(),
            OffloadKind::Pages,
        );
        kv_offload.offload(
            &text,
            1,
            keys[1][4..6].to_vec(),
            block_pool.allocate(2, 1).unwrap(),
            OffloadKind::Window {
                anchor: Some((0, keys[0][5])),
            },
        );
        let mut copies = Vec::new();
        poll_until(&mut kv_offload, |kv_offload| {
            copies.extend(kv_offload.take_offloads(&text, 1));
            copies.len() == 12
        });
        kv_offload.completed(1, true);
        poll_until(&mut kv_offload, |kv_offload| !kv_offload.offloading());

        // Every boundary past 384 lacks its window: the walk passes them.
        let mut boundary = None;
        poll_until(&mut kv_offload, |kv_offload| {
            boundary = kv_offload.admission(RequestId(2), &text, &keys, 0, 640);
            boundary.is_some()
        });
        assert_eq!(boundary, Some(384));
        assert_eq!(kv_offload.admit(RequestId(2)), Some((0, 384)));
        let loads: Vec<(u32, u32)> = kv_offload
            .loads(RequestId(2), |_, page| Some(vec![UnitId(page as u32 + 1)]))
            .iter()
            .map(|copy| (copy.group_id, copy.units[0].0 - 1))
            .collect();
        assert_eq!(
            loads,
            [
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (0, 5),
                (1, 4),
                (1, 5)
            ]
        );
    }

    /// Keys extended over the tokens a request generated are the keys of
    /// its migration's prompt, which continues with them.
    #[test]
    fn keys_extended_over_generated_tokens_are_the_migrations_keys() {
        let server = Running::start();
        let (_worker, pools) = worker_pools(&server, &[4, 5]);
        let mut kv_offload =
            KvOffloadManager::connect(&server.socket, "model", 0, DEFAULT_BOUNDARY_INTERVAL)
                .unwrap();
        let cache = report(
            vec![cache_group(None), cache_group(Some(128))],
            &pools,
            &[4, 5],
        );
        let text = WorkerGroupId("text".into());
        assert!(kv_offload.add_worker(
            &text,
            &[RankCache {
                local: true,
                cache: &cache,
            }]
        ));
        let prompt: Vec<u32> = (0..200).collect();
        let generated: Vec<u32> = (500..700).collect();
        let migrated: Vec<u32> = prompt.iter().chain(&generated).copied().collect();
        let mut keys = kv_offload.page_keys(&text, &prompt, Some(3), prompt.len());
        kv_offload.extend_page_keys(
            &text,
            &mut keys,
            CachedTokens::new(&prompt, &generated),
            Some(3),
        );
        assert_eq!(
            keys,
            kv_offload.page_keys(&text, &migrated, Some(3), migrated.len())
        );
        assert!(keys.iter().all(|group| !group.is_empty()));
    }

    /// A prompt offloads its sliding windows before every interval of pages
    /// of the largest block size, before the last boundary a reissue of it
    /// reuses, and before the last boundary of its published pages.
    #[test]
    fn sliding_windows_offload_before_aligned_boundaries() {
        let server = Running::start();
        let (_worker, pools) = worker_pools(&server, &[6, 7]);
        let mut kv_offload = KvOffloadManager::connect(&server.socket, "model", 0, 4).unwrap();
        let cache = report(
            vec![cache_group(None), cache_group(Some(128))],
            &pools,
            &[6, 7],
        );
        let text = WorkerGroupId("text".into());
        assert!(kv_offload.add_worker(
            &text,
            &[RankCache {
                local: true,
                cache: &cache,
            }]
        ));
        // Intervals of four pages of 64 tokens: 256 and 512. A reissue of
        // the 640-token prompt reuses up to 576, and its last page ends at
        // 640.
        assert_eq!(
            kv_offload.offload_boundaries(&text, 640, 0, 640),
            [256, 512, 576, 640]
        );
        // A call that computed tokens 300..600 reached two of them.
        assert_eq!(
            kv_offload.offload_boundaries(&text, 640, 300, 600),
            [512, 576]
        );
    }
}

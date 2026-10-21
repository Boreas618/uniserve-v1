//! The host page tier (L2) of the prefix cache, and the distributed tier
//! (L3) behind it, as the scheduler drives them.
//!
//! Every host's Storage Server keeps page pools of host memory; each rank
//! of a worker group maps one for its cache group and reports it
//! (`PrefixCacheInfo::page_pools`). The server's Mooncake connector, when it
//! runs one, writes every committed page to Mooncake (L3) and reads pages
//! back into the pool on request. The scheduler decides which pages move:
//!
//! - **Reads.** At admission, after the device prefix (L1) match, the pages
//!   that follow it are pinned in every rank's pool, and looked up in L2
//!   and L3. When L3 extends the run every rank's pool holds by at least
//!   [`PREFETCH_MIN_TOKENS`] tokens, those pages are prefetched into the
//!   pools; admission waits for them at most [`prefetch_wait`], then pins
//!   the pages that arrived. The leading run every rank holds is the tier's
//!   hit: admission allocates units for it, the request's first prefill
//!   batch reads the pages into them (`Batch::page_reads`) and computes from
//!   the end of the hit, and the pins end once that call's result is
//!   accepted. Pages that did not arrive in time are computed.
//! - **Writes.** Once a prompt page is published in L1, a slot is reserved
//!   for it in every rank's pool, while the scheduler holds a reference to
//!   its units. The next text batch the worker group runs copies the page
//!   (`Batch::page_writes`), and the batch's output commits it, or aborts it
//!   when the batch failed. Every published page is written (write
//!   threshold one).
//!
//! A page's key chains the page keys before it: `h_i = SHA-256(h_{i-1} ||
//! group || modality || tokens)` over full pages, each token a little-endian
//! `u32`, from `h_{-1} = SHA-256("uniserve-page" || content || isolation
//! key)`, where the content is what the ranks report for the group and the
//! isolation key keeps tenants' prefixes apart as L1 does.
//!
//! The tier serves worker groups whose cache is one full-attention group.
//! A rank's pool is reached through its host's Storage Server: on the
//! engine's host through the server's socket, on another host over the TCP
//! address the rank reports for it. Requests to the servers never block
//! the scheduler:
//! they are submitted, and their answers are taken by [`Tiers::poll`] at
//! every step; while answers are awaited, a parked scheduler wakes after a
//! millisecond to take them.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ipc::{PageCopy, PrefixCacheInfo};
use ipc::{RequestId, UnitId};
use sha2::{Digest, Sha256};
use storage::protocol::{Event, PageSlot, Reply, Request};
use storage::{Client, Key, Role, Scope};

/// The fewest tokens an L3 hit must add to the pools' run to be prefetched:
/// a shorter hit is computed faster than it is fetched.
pub(crate) const PREFETCH_MIN_TOKENS: usize = 256;

/// How long admission waits for a prefetch of `tokens` tokens: one second,
/// and a quarter second per 1024 tokens.
pub(crate) fn prefetch_wait(tokens: usize) -> Duration {
    Duration::from_secs(1) + Duration::from_millis(250 * tokens as u64 / 1024)
}

use super::PageRef;
use crate::executor::WorkerGroupId;

/// One rank's page pool.
#[derive(Clone)]
struct RankPool {
    /// The connection to the rank's host's server, in `Tiers::clients`.
    client: usize,
    scope: Scope,
    pool: u64,
}

/// The page pools of one worker group.
#[derive(Clone)]
struct GroupPools {
    ranks: Vec<RankPool>,
    content: [u8; 32],
    block_size: usize,
}

/// What a submitted request was for.
enum Purpose {
    /// Admission pins of `request` in `rank`'s pool.
    Pin { request: RequestId, rank: usize },
    /// The L2 and L3 lookup of `request`'s pages in `rank`'s pool.
    Lookup { request: RequestId, rank: usize },
    /// Pins of `request`'s prefetched pages in `rank`'s pool, at these
    /// positions of its keys.
    Repin {
        request: RequestId,
        rank: usize,
        positions: Vec<usize>,
    },
    /// Pins of `keys` in `pool` for a request that no longer wants them:
    /// the pages they pinned are unpinned when the answer arrives.
    Orphan { pool: RankPool, keys: Vec<Key> },
    /// The reservation of `write` in `rank`'s pool.
    Reserve { write: u64, rank: usize },
    /// A request whose answer changes nothing here: commit, abort, unpin,
    /// prefetch.
    Settle,
}

/// Where a request's admission stands.
#[derive(Clone, Copy)]
enum Stage {
    /// Pins and lookups are in flight.
    Pinning,
    /// The pages `from..to` are being prefetched; `arrived` ranks have
    /// reported.
    Fetching {
        from: usize,
        to: usize,
        deadline: Instant,
        arrived: usize,
    },
    /// The prefetched pages are being pinned; `awaiting` ranks have not
    /// answered.
    Repinning { awaiting: usize },
    /// The hit is known.
    Settled,
}

/// The admission pins of one request.
struct Pins {
    worker: WorkerGroupId,
    /// The first page pinned: the end of the L1 match when the pins were
    /// taken.
    start: usize,
    keys: Vec<Key>,
    /// Each rank's slots, once its pin is answered.
    slots: Vec<Option<Vec<Option<u64>>>>,
    /// Each rank's run of pages in L2 or L3, once its lookup is answered.
    l3: Vec<Option<usize>>,
    stage: Stage,
}

/// A request's page hit of the tier, from its admission until its reads
/// are accepted.
struct Hit {
    /// The first page of the hit.
    start: usize,
    worker: WorkerGroupId,
    keys: Vec<Key>,
    /// Each rank's slot of each page.
    slots: Vec<Vec<u64>>,
    /// Whether a batch carries the reads.
    sent: bool,
}

/// The reservations of one group of published pages.
struct Reservation {
    worker: WorkerGroupId,
    keys: Vec<Key>,
    pages: Vec<Arc<PageRef>>,
    answered: usize,
}

/// One page write waiting for, or carried by, a batch.
struct Write {
    /// The connection to the pool's server.
    client: usize,
    copy: PageCopy,
    key: Key,
    /// Holds the page's units until the write is settled.
    _page: Arc<PageRef>,
}

/// One rank's report for [`Tiers::add_worker`].
pub(crate) struct RankCache<'a> {
    /// Whether the rank runs on the engine's host.
    pub local: bool,
    pub cache: &'a PrefixCacheInfo,
}

/// The scheduler's side of the host page tier.
pub(crate) struct Tiers {
    deployment: String,
    /// The connections to the servers: the engine's host's first, then other
    /// hosts' by the TCP address they were reached at.
    clients: Vec<Client>,
    remote: HashMap<String, usize>,
    groups: HashMap<WorkerGroupId, GroupPools>,
    /// Requests submitted and not yet answered, by connection and request.
    pending: HashMap<(usize, u64), Purpose>,
    pins: HashMap<RequestId, Pins>,
    /// The request each prefetch in flight is for, by tag.
    fetches: HashMap<u64, RequestId>,
    next_tag: u64,
    hits: HashMap<RequestId, Hit>,
    reservations: HashMap<u64, Reservation>,
    next_reservation: u64,
    /// Writes reserved and waiting for a batch of their worker group.
    queued: HashMap<WorkerGroupId, Vec<Write>>,
    /// Writes carried by submitted batches, by batch, with the worker group
    /// that runs it.
    carried: HashMap<u64, (WorkerGroupId, Vec<Write>)>,
}

impl Tiers {
    /// Connect to this host's Storage Server, at `socket`, as a head of
    /// `deployment`.
    pub(crate) fn connect(socket: &std::path::Path, deployment: &str) -> anyhow::Result<Self> {
        let client = Client::connect(
            socket,
            Role::Head {
                deployment: deployment.to_owned(),
            },
        )?;
        Ok(Self {
            deployment: deployment.to_owned(),
            clients: vec![client],
            remote: HashMap::new(),
            groups: HashMap::new(),
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
        };
        match Client::connect_tcp(address, role) {
            Ok(client) => {
                self.clients.push(client);
                self.remote
                    .insert(address.to_owned(), self.clients.len() - 1);
                Some(self.clients.len() - 1)
            }
            Err(error) => {
                tracing::warn!(address, %error, "the host page tier cannot reach a storage server");
                None
            }
        }
    }

    /// Serve `worker`, whose ranks (`ranks`, in rank order) are the scopes
    /// of the deployment named by the group and their ranks, when every rank
    /// reports a page pool for one full-attention cache group and its server
    /// is reachable. Returns whether the tier serves it.
    pub(crate) fn add_worker(&mut self, worker: &WorkerGroupId, ranks: &[RankCache<'_>]) -> bool {
        let Some(first) = ranks.first() else {
            return false;
        };
        let [group] = &first.cache.groups[..] else {
            return false;
        };
        if group.kind.window().is_some() {
            return false;
        }
        let mut pools = Vec::with_capacity(ranks.len());
        let mut content = None;
        for (rank, report) in ranks.iter().enumerate() {
            let [pool] = &report.cache.page_pools[..] else {
                return false;
            };
            if pool.group_id != 0 || content.is_some_and(|content| content != pool.content) {
                return false;
            }
            content = Some(pool.content);
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
            pools.push(RankPool {
                client,
                scope: Scope {
                    deployment: self.deployment.clone(),
                    group: worker.0.clone(),
                    rank: rank as u32,
                },
                pool: pool.pool,
            });
        }
        let ranks = pools;
        let Some(content) = content else {
            return false;
        };
        self.groups.insert(
            worker.clone(),
            GroupPools {
                ranks,
                content,
                block_size: group.block_size as usize,
            },
        );
        tracing::info!(worker = %worker.0, "the host page tier serves the worker group");
        true
    }

    /// Whether the tier serves `worker`.
    pub(crate) fn serves(&self, worker: &WorkerGroupId) -> bool {
        self.groups.contains_key(worker)
    }

    /// The tokens of one page of `worker`'s cache group.
    pub(crate) fn block_size(&self, worker: &WorkerGroupId) -> Option<usize> {
        self.groups.get(worker).map(|group| group.block_size)
    }

    /// The keys of the first `pages` full pages of `prompt` for `worker`.
    pub(crate) fn page_keys(
        &self,
        worker: &WorkerGroupId,
        prompt: &[u32],
        isolation_key: Option<u64>,
        pages: usize,
    ) -> Vec<Key> {
        let Some(group) = self.groups.get(worker) else {
            return Vec::new();
        };
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
        let mut chain: [u8; 32] = digest.finalize().into();
        let mut keys = Vec::with_capacity(pages);
        for tokens in prompt.chunks_exact(group.block_size).take(pages) {
            let mut digest = Sha256::new();
            digest.update(chain);
            // The only cache group, and text (`TEXT_MODALITY_TAG`).
            digest.update(0_u32.to_le_bytes());
            digest.update([0_u8]);
            for token in tokens {
                digest.update(token.to_le_bytes());
            }
            chain = digest.finalize().into();
            keys.push(Key(chain));
        }
        keys
    }

    fn submit(&mut self, client: usize, request: &Request, purpose: Purpose) -> bool {
        match self.clients[client].submit(request, &[]) {
            Ok(id) => {
                self.pending.insert((client, id), purpose);
                true
            }
            Err(error) => {
                tracing::warn!(%error, "a host page tier request failed");
                false
            }
        }
    }

    /// The tier's answer for admitting `request` to `worker` with `l1_pages`
    /// pages cached in L1: `None` while pins, lookups or prefetches are in
    /// flight, else the hit (possibly empty), which the caller takes with
    /// [`Tiers::admit`]. The first call pins `keys[l1_pages..]` (the
    /// request's reusable full pages) in every rank's pool and looks them up
    /// in L2 and L3. Pins taken for another L1 match are released and taken
    /// again.
    pub(crate) fn admission(
        &mut self,
        request: RequestId,
        worker: &WorkerGroupId,
        keys: &[Key],
        l1_pages: usize,
    ) -> Option<usize> {
        if let Some(pins) = self.pins.get(&request) {
            if pins.start != l1_pages || pins.worker != *worker {
                self.release(request);
            } else {
                return self.advance(request);
            }
        }
        let Some(group) = self.groups.get(worker).cloned() else {
            return Some(0);
        };
        let wanted: Vec<Key> = keys.get(l1_pages..).unwrap_or_default().to_vec();
        if wanted.is_empty() {
            return Some(0);
        }
        // L3 can only matter when the pages could reach a prefetch.
        let look_up = wanted.len() * group.block_size >= PREFETCH_MIN_TOKENS;
        let ranks = group.ranks.len();
        let mut pins = Pins {
            worker: worker.clone(),
            start: l1_pages,
            keys: wanted.clone(),
            slots: vec![None; ranks],
            l3: vec![if look_up { None } else { Some(0) }; ranks],
            stage: Stage::Pinning,
        };
        for (rank, pool) in group.ranks.iter().enumerate() {
            let pinned = self.submit(
                pool.client,
                &Request::Pin {
                    pool: pool.pool,
                    reader: pool.scope.clone(),
                    keys: wanted.clone(),
                },
                Purpose::Pin { request, rank },
            );
            if !pinned {
                // A rank without pins holds nothing: the hit is empty.
                pins.slots[rank] = Some(Vec::new());
            }
            if look_up {
                let looked = self.submit(
                    pool.client,
                    &Request::Lookup {
                        pool: pool.pool,
                        keys: wanted.clone(),
                    },
                    Purpose::Lookup { request, rank },
                );
                if !looked {
                    pins.l3[rank] = Some(0);
                }
            }
        }
        self.pins.insert(request, pins);
        None
    }

    /// Move `request`'s admission on from its current stage; see
    /// [`Tiers::admission`].
    fn advance(&mut self, request: RequestId) -> Option<usize> {
        let pins = self.pins.get(&request)?;
        let group = self.groups.get(&pins.worker)?.clone();
        match pins.stage {
            Stage::Settled => Some(Self::run(pins)),
            Stage::Repinning { awaiting } => {
                if awaiting > 0 {
                    return None;
                }
                let pins = self.pins.get_mut(&request)?;
                pins.stage = Stage::Settled;
                Some(Self::run(pins))
            }
            Stage::Pinning => {
                let answered =
                    pins.slots.iter().all(Option::is_some) && pins.l3.iter().all(Option::is_some);
                if !answered {
                    return None;
                }
                let run = Self::run(pins);
                let l3 = pins
                    .l3
                    .iter()
                    .flatten()
                    .copied()
                    .min()
                    .unwrap_or(0)
                    .min(pins.keys.len());
                if l3 <= run || (l3 - run) * group.block_size < PREFETCH_MIN_TOKENS {
                    let pins = self.pins.get_mut(&request)?;
                    pins.stage = Stage::Settled;
                    return Some(run);
                }
                // Every rank's pool fetches the pages its L3 holds past the
                // run; pages a pool holds already are left as they are.
                let keys = pins.keys[run..l3].to_vec();
                let mut arrived = 0;
                for pool in &group.ranks {
                    let tag = self.next_tag;
                    self.next_tag += 1;
                    let submitted = self.submit(
                        pool.client,
                        &Request::Prefetch {
                            pool: pool.pool,
                            keys: keys.clone(),
                            tag,
                        },
                        Purpose::Settle,
                    );
                    if submitted {
                        self.fetches.insert(tag, request);
                    } else {
                        arrived += 1;
                    }
                }
                let pins = self.pins.get_mut(&request)?;
                pins.stage = Stage::Fetching {
                    from: run,
                    to: l3,
                    deadline: Instant::now() + prefetch_wait((l3 - run) * group.block_size),
                    arrived,
                };
                None
            }
            Stage::Fetching {
                from,
                to,
                deadline,
                arrived,
            } => {
                if arrived < group.ranks.len() && Instant::now() < deadline {
                    return None;
                }
                // Pin what arrived; pages still in flight are computed, and
                // stay in the pools for later requests.
                self.fetches.retain(|_, waiting| *waiting != request);
                let (keys, ranks) = (pins.keys.clone(), pins.slots.clone());
                let mut awaiting = 0;
                for (rank, pool) in group.ranks.iter().enumerate() {
                    let Some(slots) = ranks[rank].as_ref() else {
                        continue;
                    };
                    let positions: Vec<usize> = (from..to)
                        .filter(|position| slots.get(*position).is_some_and(Option::is_none))
                        .collect();
                    if positions.is_empty() {
                        continue;
                    }
                    let keys = positions.iter().map(|position| keys[*position]).collect();
                    if self.submit(
                        pool.client,
                        &Request::Pin {
                            pool: pool.pool,
                            reader: pool.scope.clone(),
                            keys,
                        },
                        Purpose::Repin {
                            request,
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
                    return Some(Self::run(pins));
                }
                pins.stage = Stage::Repinning { awaiting };
                None
            }
        }
    }

    /// The leading run of pages every rank pinned.
    fn run(pins: &Pins) -> usize {
        pins.slots
            .iter()
            .map(|slots| {
                slots.as_ref().map_or(0, |slots| {
                    slots.iter().take_while(|slot| slot.is_some()).count()
                })
            })
            .min()
            .unwrap_or(0)
    }

    /// Admit `request` with its pinned pages: returns the hit's first page
    /// and its pages. The pins past the hit end now; the hit's end when its
    /// reads are done ([`Tiers::read_done`]) or the request leaves
    /// ([`Tiers::sweep`]).
    pub(crate) fn admit(&mut self, request: RequestId) -> Option<(usize, usize)> {
        let pins = self.pins.remove(&request)?;
        let pages = Self::run(&pins);
        let group = self.groups.get(&pins.worker)?.clone();
        let mut slots = Vec::with_capacity(group.ranks.len());
        for (rank, pool) in group.ranks.iter().enumerate() {
            let pinned = pins.slots[rank].clone().unwrap_or_default();
            let extra: Vec<Key> = pins
                .keys
                .iter()
                .zip(&pinned)
                .skip(pages)
                .filter(|(_, slot)| slot.is_some())
                .map(|(key, _)| *key)
                .collect();
            if !extra.is_empty() {
                self.unpin(pool, extra);
            }
            slots.push(pinned.iter().take(pages).flatten().copied().collect());
        }
        if pages == 0 {
            return None;
        }
        let start = pins.start;
        self.hits.insert(
            request,
            Hit {
                start,
                worker: pins.worker,
                keys: pins.keys[..pages].to_vec(),
                slots,
                sent: false,
            },
        );
        Some((start, pages))
    }

    /// Release `request`'s admission pins: it left the queue, or its L1
    /// match changed.
    pub(crate) fn release(&mut self, request: RequestId) {
        let Some(pins) = self.pins.remove(&request) else {
            return;
        };
        self.fetches.retain(|_, waiting| *waiting != request);
        let Some(group) = self.groups.get(&pins.worker).cloned() else {
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
                    rank,
                } if *pinned == request => Some((*rank, pins.keys.clone())),
                Purpose::Repin {
                    request: pinned,
                    rank,
                    positions,
                } if *pinned == request => Some((
                    *rank,
                    positions
                        .iter()
                        .map(|position| pins.keys[*position])
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
            if let Some((rank, keys)) = orphan
                && let Some(pool) = group.ranks.get(rank)
            {
                *purpose = Purpose::Orphan {
                    pool: pool.clone(),
                    keys,
                };
            }
        }
        for (rank, pool) in group.ranks.iter().enumerate() {
            let pinned = Self::pinned(&pins.keys, pins.slots[rank].as_deref());
            if !pinned.is_empty() {
                self.unpin(pool, pinned);
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

    /// The page reads of `request`'s hit, once: `units(page)` names the
    /// units of an absolute page of the request's table, freshly allocated
    /// for the hit. Empty for a request without a hit or whose reads a
    /// batch already carries.
    pub(crate) fn reads(
        &mut self,
        request: RequestId,
        units: impl Fn(usize) -> Option<Vec<UnitId>>,
    ) -> Vec<PageCopy> {
        let Some(hit) = self.hits.get_mut(&request).filter(|hit| !hit.sent) else {
            return Vec::new();
        };
        let Some(group) = self.groups.get(&hit.worker) else {
            return Vec::new();
        };
        hit.sent = true;
        let mut copies = Vec::new();
        for (rank, pool) in group.ranks.iter().enumerate() {
            for (index, slot) in hit.slots[rank].iter().enumerate() {
                if let Some(units) = units(hit.start + index) {
                    copies.push(PageCopy {
                        rank: rank as u32,
                        pool: pool.pool,
                        group_id: 0,
                        slot: *slot,
                        units,
                    });
                }
            }
        }
        copies
    }

    /// The pages of `request`'s hit, which hold the tier's bytes already.
    pub(crate) fn hit_pages(&self, request: RequestId) -> std::ops::Range<usize> {
        self.hits
            .get(&request)
            .map_or(0..0, |hit| hit.start..hit.start + hit.keys.len())
    }

    /// `request`'s reads were accepted: its pins end.
    pub(crate) fn read_done(&mut self, request: RequestId) {
        if self.hits.get(&request).is_some_and(|hit| hit.sent) {
            self.drop_hit(request);
        }
    }

    fn drop_hit(&mut self, request: RequestId) {
        let Some(hit) = self.hits.remove(&request) else {
            return;
        };
        let Some(group) = self.groups.get(&hit.worker).cloned() else {
            return;
        };
        for pool in &group.ranks {
            self.unpin(pool, hit.keys.clone());
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

    /// Write published pages of `worker` to the tier: `keys[i]` is the key
    /// of the page `pages[i]` references. A slot is reserved for each page
    /// in every rank's pool; a reserved page waits for a batch of `worker`.
    pub(crate) fn write(&mut self, worker: &WorkerGroupId, keys: Vec<Key>, pages: Vec<PageRef>) {
        let Some(group) = self.groups.get(worker).cloned() else {
            return;
        };
        if keys.is_empty() || keys.len() != pages.len() {
            return;
        }
        let id = self.next_reservation;
        self.next_reservation += 1;
        let mut submitted = 0;
        for (rank, pool) in group.ranks.iter().enumerate() {
            if self.submit(
                pool.client,
                &Request::Reserve {
                    pool: pool.pool,
                    writer: pool.scope.clone(),
                    keys: keys.clone(),
                },
                Purpose::Reserve { write: id, rank },
            ) {
                submitted += 1;
            }
        }
        if submitted > 0 {
            self.reservations.insert(
                id,
                Reservation {
                    worker: worker.clone(),
                    keys,
                    pages: pages.into_iter().map(Arc::new).collect(),
                    answered: group.ranks.len() - submitted,
                },
            );
        }
    }

    /// Whether published pages are still on their way to the tier: slots
    /// being reserved, writes waiting for or carried by a batch, or their
    /// commits unanswered.
    pub(crate) fn writing(&self) -> bool {
        !self.reservations.is_empty()
            || self.queued.values().any(|writes| !writes.is_empty())
            || !self.carried.is_empty()
            || self
                .pending
                .values()
                .any(|purpose| matches!(purpose, Purpose::Reserve { .. } | Purpose::Settle))
    }

    /// The worker groups whose reserved page writes wait for a batch.
    pub(crate) fn queued_workers(&self) -> Vec<WorkerGroupId> {
        self.queued
            .iter()
            .filter(|(_, writes)| !writes.is_empty())
            .map(|(worker, _)| worker.clone())
            .collect()
    }

    /// The page writes queued for `worker`, which the batch `batch` carries.
    pub(crate) fn take_writes(&mut self, worker: &WorkerGroupId, batch: u64) -> Vec<PageCopy> {
        let Some(writes) = self.queued.remove(worker) else {
            return Vec::new();
        };
        let copies: Vec<PageCopy> = writes.iter().map(|write| write.copy.clone()).collect();
        tracing::debug!(
            worker = %worker.0,
            batch,
            pages = copies.len(),
            "a batch carries page writes to the host page tier"
        );
        self.carried
            .entry(batch)
            .or_insert_with(|| (worker.clone(), Vec::new()))
            .1
            .extend(writes);
        copies
    }

    /// The batch `batch` completed: the pages it wrote are committed when
    /// every call of it succeeded, and aborted otherwise.
    pub(crate) fn completed(&mut self, batch: u64, succeeded: bool) {
        let Some((_, writes)) = self.carried.remove(&batch) else {
            return;
        };
        self.settle(writes, succeeded);
    }

    fn settle(&mut self, writes: Vec<Write>, succeeded: bool) {
        let mut by_pool: HashMap<(usize, u64), Vec<Key>> = HashMap::new();
        for write in &writes {
            by_pool
                .entry((write.client, write.copy.pool))
                .or_default()
                .push(write.key);
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

    /// `worker`'s ranks were replaced: the writes waiting for its batches
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
            if let Some((_, writes)) = self.carried.remove(&batch) {
                lost.extend(writes);
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
                tracing::warn!(%error, "a host page tier connection failed");
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
                (Purpose::Pin { request, rank }, reply) => {
                    let slots = match reply {
                        Reply::Pinned { slots } => slots,
                        other => {
                            tracing::warn!(?other, "a host page tier pin failed");
                            Vec::new()
                        }
                    };
                    if let Some(pins) = self.pins.get_mut(&request) {
                        pins.slots[rank] = Some(slots);
                        admissions = true;
                    }
                }
                (Purpose::Lookup { request, rank }, reply) => {
                    let l3 = match reply {
                        Reply::Lookup { l3, .. } => l3 as usize,
                        other => {
                            tracing::warn!(?other, "a host page tier lookup failed");
                            0
                        }
                    };
                    if let Some(pins) = self.pins.get_mut(&request) {
                        pins.l3[rank] = Some(l3);
                        admissions = true;
                    }
                }
                (
                    Purpose::Repin {
                        request,
                        rank,
                        positions,
                    },
                    reply,
                ) => {
                    let slots = match reply {
                        Reply::Pinned { slots } => slots,
                        other => {
                            tracing::warn!(?other, "a host page tier pin failed");
                            Vec::new()
                        }
                    };
                    if let Some(pins) = self.pins.get_mut(&request) {
                        if let Some(pinned) = pins.slots[rank].as_mut() {
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
                (Purpose::Reserve { write, rank }, reply) => {
                    let slots = match reply {
                        Reply::Reserved { slots } => slots,
                        other => {
                            tracing::warn!(?other, "a host page tier reservation failed");
                            Vec::new()
                        }
                    };
                    self.reserved(write, rank, &slots);
                }
                (Purpose::Settle, Reply::Error { kind, message }) => {
                    tracing::warn!(?kind, message, "a host page tier request failed");
                }
                (Purpose::Settle, _) => {}
            }
        }
        admissions
    }

    /// `rank` answered reservation `id` with `slots`: its reserved pages
    /// wait for a batch.
    fn reserved(&mut self, id: u64, rank: usize, slots: &[PageSlot]) {
        let Some(reservation) = self.reservations.get_mut(&id) else {
            return;
        };
        let pool = self
            .groups
            .get(&reservation.worker)
            .and_then(|group| group.ranks.get(rank))
            .map(|pool| (pool.client, pool.pool));
        if let Some((client, pool)) = pool {
            for ((key, page), slot) in reservation.keys.iter().zip(&reservation.pages).zip(slots) {
                if let PageSlot::Reserved { slot } = slot {
                    self.queued
                        .entry(reservation.worker.clone())
                        .or_default()
                        .push(Write {
                            client,
                            copy: PageCopy {
                                rank: rank as u32,
                                pool,
                                group_id: 0,
                                slot: *slot,
                                units: page.units().to_vec(),
                            },
                            key: *key,
                            _page: Arc::clone(page),
                        });
                }
            }
        }
        reservation.answered += 1;
        let done = self
            .groups
            .get(&reservation.worker)
            .is_none_or(|group| reservation.answered >= group.ranks.len());
        if done {
            self.reservations.remove(&id);
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
                let mut server = Server::new(listener, 64 * 16, None).unwrap();
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

    #[test]
    fn admission_prefetches_a_long_distributed_hit_and_pins_what_arrived() {
        let server = Running::start();
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
            },
        )
        .unwrap();
        let shape = PageShape {
            content: Key([3; 32]),
            kv_head_offset: 0,
            kv_head_count: 8,
            page_bytes: 16,
        };
        let request = Request::PagePool {
            shape,
            numa_node: None,
        };
        let Reply::PagePool { pool, .. } = worker.call(&request, &[]).unwrap().reply else {
            panic!("the page pool was refused");
        };
        let stop = Arc::new(AtomicBool::new(false));
        let mooncake = connector(&server.socket, Arc::clone(&stop));

        let mut tiers = Tiers::connect(&server.socket, "model").unwrap();
        let cache = PrefixCacheInfo {
            num_units: 64,
            dtype: PrefixCacheDtype::BFloat16,
            groups: vec![CacheGroup {
                kind: CacheGroupKind::Full,
                block_size: 64,
                units_per_page: 1,
                layer_ids: vec![0],
                num_kv_heads: 8,
                total_kv_heads: 8,
                kv_head_offset: 0,
                head_dim: 128,
            }],
            page_pools: vec![PagePoolInfo {
                group_id: 0,
                pool,
                content: [3; 32],
                endpoint: None,
            }],
            scratch_units: 0,
        };
        let text = WorkerGroupId("text".into());
        assert!(tiers.add_worker(
            &text,
            &[RankCache {
                local: true,
                cache: &cache,
            }]
        ));

        // Eight pages that only L3 holds: 512 tokens are worth a prefetch,
        // and every page arrives within the wait.
        let prompt: Vec<u32> = (0..8 * 64 + 1).collect();
        let keys = tiers.page_keys(&text, &prompt, None, 8);
        let deadline = Instant::now() + Duration::from_secs(10);
        let pages = loop {
            tiers.poll();
            if let Some(pages) = tiers.admission(RequestId(1), &text, &keys, 0) {
                break pages;
            }
            assert!(Instant::now() < deadline, "the admission did not settle");
            std::thread::sleep(Duration::from_millis(1));
        };
        assert_eq!(pages, 8);
        assert_eq!(tiers.admit(RequestId(1)), Some((0, 8)));

        stop.store(true, Ordering::Release);
        mooncake.join().unwrap();
    }
}

//! The host tier of requests' encoder outputs, as the scheduler drives it.
//!
//! A request's encoder outputs are derived state: they follow from what the
//! encoders read, the request's facts, and from what computes them. A video
//! request's outputs are its prompt's text conditioning, the vision features
//! of its references and its condition latents; an image request's are the
//! vision and latent features of each image it reads, its input images and
//! the images it generated and feeds back. Every rank that executes an
//! encoder maps its host's encoder pool (`ipc::EncoderPoolInfo`), shared by
//! the deployment's ranks on that host in every world of the deployment. An
//! output is stored whole, its rows in order, in the pages its bytes fill; a
//! page's key combines its output's content key and page index. The output
//! key uses the shared storage key builder over:
//! the output's name, the identity of what computes it (the weights' values
//! and the declared function and format identities of the encoder's
//! modules, `EncoderPoolInfo::identities`) and the digest of what the
//! encoder read (`proto::request::EncoderInputs` for a video request, which
//! the daemon computes from the request's facts; for an image request
//! `crate::scheduler::encodings`, over the image's digest and, for an
//! encoder that samples, the request's draw). Requests that read the same in
//! worlds that compute it the same, a resumed request among them, so find
//! each other's outputs, and a world that computes otherwise finds none.
//!
//! - **Lookup.** Every page of an output is pinned before its encoder calls
//!   are planned. When every page is pinned, the output's calls read their
//!   rows from the pages instead of computing them (`Call::host_output`, an
//!   import), and the pins end once the calls completed.
//! - **Store.** When a page is missing, the pages pinned are released and
//!   every page is reserved: the encoder calls compute the output and write
//!   the rows they wrote to the reserved pages (an export), which are
//!   committed once every call of the output completed, and aborted when
//!   one failed. A page the pool holds already is left as it is.
//! - When the store cannot ask a server or a page cannot be reserved, the
//!   encoders compute the output and nothing is stored. A request whose
//!   cache policy reads no cached output starts at the reservation, and one
//!   that writes none computes a missing output without storing it.
//!
//! The scheduler plans an output's encoder calls only once its lookup
//! resolved, so all of them import, or all compute. Like the host latent
//! pools, the store connects as a head of its engine's world, so what a
//! world on trial stores stays its own until its commit, and requests to the
//! servers never block the scheduler.

use std::collections::{BTreeMap, HashMap};

use ipc::{CallId, EncoderPoolInfo, HostOutput, RequestKey};
use storage::key::Digest;
use storage::protocol::{PageSlot, Reply, Request};
use storage::{Key, Scope};

use crate::executor::WorkerGroupId;
use crate::host_tier::Servers;

/// One encoder output of a request: its name (the product an encoder
/// writes) and its ordinal among the request's outputs of that name. A video
/// request has one output of each name; an image request one per image
/// encoding, in the order it encodes them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct OutputId {
    pub name: &'static str,
    pub ordinal: u32,
}

/// Whether a lookup may read a stored output and store a computed one, as
/// the request's cache policy allows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Access {
    pub read: bool,
    pub write: bool,
}

impl Access {
    /// Read stored outputs and store computed ones.
    pub(crate) const FULL: Self = Self {
        read: true,
        write: true,
    };
}

/// The encoder pool a worker group's ranks share.
#[derive(Clone)]
struct GroupPool {
    worker: WorkerGroupId,
    /// The connection to the pool's server (`EncoderStore::servers`).
    client: usize,
    pool: u64,
    page_bytes: u64,
    /// The group's first rank, which pins and reserves the pages for the
    /// group's ranks.
    scope: Scope,
    /// The identity of what computes each output the group writes.
    identities: BTreeMap<String, String>,
}

/// One rank's report for [`EncoderStore::add_worker`].
pub(crate) struct RankEncoders<'a> {
    /// Whether the rank runs on the engine's host.
    pub local: bool,
    pub pool: &'a EncoderPoolInfo,
}

/// The key of page `page` of an output whose content key is `content`.
fn page_key(content: &[u8; 32], page: u32) -> Key {
    let mut digest = Digest::new(b"encoder-page");
    digest.bytes(content);
    digest.number(u64::from(page));
    Key(digest.finish())
}

/// The content key of output `name`, computed by `identity` from what
/// `inputs` digests.
fn content(name: &str, identity: &str, inputs: &[u8; 32]) -> [u8; 32] {
    let mut digest = Digest::new(b"encoder-output");
    digest.text(name);
    digest.text(identity);
    digest.bytes(inputs);
    digest.finish()
}

/// Where an output's lookup stands.
enum Stage {
    /// Its pages' pins are submitted.
    Pinning,
    /// A page was missing, or the request reads no stored output: its
    /// pages' reservations are submitted.
    Reserving,
    /// Every page is pinned: the calls read the output.
    Import(Vec<u64>),
    /// The calls compute the output and write it to the reserved pages,
    /// `None` for a page the pool held already.
    Export(Vec<Option<u64>>),
    /// The calls compute the output, which is not stored.
    Compute,
}

struct Output {
    pool: GroupPool,
    keys: Vec<Key>,
    row_bytes: u64,
    /// Whether a missing output is stored once computed.
    write: bool,
    stage: Stage,
    /// The output's calls in flight.
    calls: u32,
    /// Whether the scheduler planned every call of the output.
    planned: bool,
    /// Whether one of its calls failed.
    failed: bool,
}

enum Purpose {
    /// The pins of an output's pages.
    Pin {
        request: RequestKey,
        output: OutputId,
    },
    /// The reservation of an output's pages.
    Reserve {
        request: RequestKey,
        output: OutputId,
    },
    /// Pins answered for a lookup that ended meanwhile: whatever they pinned
    /// is released.
    Orphan { pool: GroupPool, keys: Vec<Key> },
    /// A reservation answered for a lookup that ended meanwhile: whatever it
    /// reserved is aborted.
    Abandoned { pool: GroupPool, keys: Vec<Key> },
    /// An unpin, a commit or an abort, whose answer only reports failures.
    Settle,
}

/// The scheduler's side of the host encoder pools.
pub(crate) struct EncoderStore {
    servers: Servers<Purpose>,
    /// The pool of each worker group the store serves.
    workers: HashMap<WorkerGroupId, GroupPool>,
    /// Each admission's outputs.
    requests: HashMap<RequestKey, HashMap<OutputId, Output>>,
    /// The output each encoder call in flight reads or writes.
    calls: HashMap<(RequestKey, CallId), OutputId>,
    /// The outputs whose lookups ended while calls of theirs were in flight
    /// (their admission left, or the scheduler no longer needs them), which
    /// may still read or write their pages: they end once the calls
    /// completed, or once their worker group was lost.
    left: HashMap<(RequestKey, OutputId), Output>,
}

impl EncoderStore {
    /// Connect to this host's Storage Server, at `socket`, as a head of
    /// `deployment` in `world`.
    pub(crate) fn connect(
        socket: &std::path::Path,
        deployment: &str,
        world: u64,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            servers: Servers::connect(socket, deployment, world, "host encoder pools")?,
            workers: HashMap::new(),
            requests: HashMap::new(),
            calls: HashMap::new(),
            left: HashMap::new(),
        })
    }

    /// Serve `worker`, whose ranks executing encoder calls (`ranks`, each
    /// with its rank index in the group) report their host's encoder pool,
    /// when they all report the same one and its server is reachable: the
    /// group's calls read and write the same pages on every rank. An output
    /// two ranks report other identities of is not stored. Returns whether
    /// the store serves it.
    pub(crate) fn add_worker(
        &mut self,
        worker: &WorkerGroupId,
        ranks: &[(u32, RankEncoders<'_>)],
    ) -> bool {
        let Some((first_rank, first)) = ranks.first() else {
            return false;
        };
        if ranks.iter().any(|(_, report)| {
            report.pool.pool != first.pool.pool
                || report.pool.page_bytes != first.pool.page_bytes
                || report.pool.endpoint != first.pool.endpoint
                || report.local != first.local
        }) || first.pool.page_bytes == 0
        {
            return false;
        }
        let Some(client) = self
            .servers
            .client(first.local, first.pool.endpoint.as_deref())
        else {
            return false;
        };
        let mut identities = BTreeMap::new();
        let mut conflicting = Vec::new();
        for (_, report) in ranks {
            for (output, identity) in &report.pool.identities {
                match identities.get(output) {
                    Some(known) if known != identity => conflicting.push(output.clone()),
                    _ => {
                        identities.insert(output.clone(), identity.clone());
                    }
                }
            }
        }
        for output in conflicting {
            identities.remove(&output);
        }
        tracing::info!(
            worker = %worker.0,
            pool = first.pool.pool,
            outputs = identities.len(),
            "the host encoder pool serves the worker group"
        );
        self.workers.insert(
            worker.clone(),
            GroupPool {
                worker: worker.clone(),
                client,
                pool: first.pool.pool,
                page_bytes: first.pool.page_bytes,
                scope: Scope {
                    deployment: self.servers.deployment().to_owned(),
                    group: worker.0.clone(),
                    rank: *first_rank,
                },
                identities,
            },
        );
        true
    }

    /// Whether `worker`'s pool stores outputs named `name`: the store serves
    /// the group and its ranks agree on what computes them.
    pub(crate) fn serves(&self, worker: &WorkerGroupId, name: &str) -> bool {
        self.workers
            .get(worker)
            .is_some_and(|pool| pool.identities.contains_key(name))
    }

    /// Look up `output` of admission `request`, which `worker` computes from
    /// what `inputs` digests: at most `rows` rows of `row_bytes` bytes, as
    /// `access` allows. Returns whether the lookup started; without one the
    /// output's calls compute it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn lookup(
        &mut self,
        request: RequestKey,
        worker: &WorkerGroupId,
        output: OutputId,
        inputs: &[u8; 32],
        rows: u32,
        row_bytes: u64,
        access: Access,
    ) -> bool {
        if !access.read && !access.write {
            return false;
        }
        let Some(pool) = self.workers.get(worker) else {
            return false;
        };
        let Some(identity) = pool.identities.get(output.name) else {
            return false;
        };
        let bytes = u64::from(rows) * row_bytes;
        let Ok(pages) = u32::try_from(bytes.div_ceil(pool.page_bytes)) else {
            return false;
        };
        if pages == 0 {
            return false;
        }
        let content = content(output.name, identity, inputs);
        let keys: Vec<Key> = (0..pages).map(|page| page_key(&content, page)).collect();
        let pool = pool.clone();
        // A request that reads no stored output stores its own at once.
        let (submitted, stage) = if access.read {
            let pin = Request::Pin {
                pool: pool.pool,
                reader: pool.scope.clone(),
                keys: keys.clone(),
            };
            let purpose = Purpose::Pin { request, output };
            (
                self.servers.submit(pool.client, &pin, purpose),
                Stage::Pinning,
            )
        } else {
            let reserve = Request::Reserve {
                pool: pool.pool,
                writer: pool.scope.clone(),
                keys: keys.clone(),
                window: None,
            };
            let purpose = Purpose::Reserve { request, output };
            (
                self.servers.submit(pool.client, &reserve, purpose),
                Stage::Reserving,
            )
        };
        if !submitted {
            return false;
        }
        self.requests.entry(request).or_default().insert(
            output,
            Output {
                pool,
                keys,
                row_bytes,
                write: access.write,
                stage,
                calls: 0,
                planned: false,
                failed: false,
            },
        );
        true
    }

    /// Whether `request`'s `output` may be computed or read: its lookup
    /// resolved, or it has none.
    pub(crate) fn resolved(&self, request: RequestKey, output: OutputId) -> bool {
        self.requests
            .get(&request)
            .and_then(|outputs| outputs.get(&output))
            .is_none_or(|output| !matches!(output.stage, Stage::Pinning | Stage::Reserving))
    }

    /// The host output of `call` of `request`'s resolved `output`, whose
    /// output holds the stored output's rows `rows`, and the call counted in
    /// flight; `last` when no other call of the output follows. `None` for
    /// an output the store neither reads nor writes.
    pub(crate) fn plan_call(
        &mut self,
        request: RequestKey,
        output: OutputId,
        call: CallId,
        rows: [u32; 2],
        last: bool,
    ) -> Option<HostOutput> {
        let stored = self.requests.get_mut(&request)?.get_mut(&output)?;
        let (import, slots) = match &stored.stage {
            Stage::Import(slots) => (true, slots.iter().copied().map(Some).collect()),
            Stage::Export(slots) => (false, slots.clone()),
            Stage::Pinning | Stage::Reserving | Stage::Compute => return None,
        };
        stored.calls += 1;
        stored.planned |= last;
        let host = HostOutput {
            pool: stored.pool.pool,
            import,
            row_bytes: stored.row_bytes,
            rows,
            slots,
        };
        self.calls.insert((request, call), output);
        Some(host)
    }

    /// `call` of `request` completed, which `succeeded` or not. Once every
    /// call of the output it read or wrote completed, an imported output's
    /// pins end, and an exported one's pages are committed when every call
    /// succeeded and aborted otherwise. A call of no output is ignored.
    pub(crate) fn call_completed(&mut self, request: RequestKey, call: CallId, succeeded: bool) {
        let Some(output) = self.calls.remove(&(request, call)) else {
            return;
        };
        // An output whose lookup ended ends with its last call.
        if let Some(stored) = self.left.get_mut(&(request, output)) {
            stored.calls = stored.calls.saturating_sub(1);
            if stored.calls == 0
                && let Some(stored) = self.left.remove(&(request, output))
            {
                self.settle(stored);
            }
            return;
        }
        let Some(outputs) = self.requests.get_mut(&request) else {
            return;
        };
        let Some(stored) = outputs.get_mut(&output) else {
            return;
        };
        stored.calls = stored.calls.saturating_sub(1);
        stored.failed |= !succeeded;
        if stored.calls > 0 || !stored.planned {
            return;
        }
        let Some(stored) = outputs.remove(&output) else {
            return;
        };
        if outputs.is_empty() {
            self.requests.remove(&request);
        }
        self.settle(stored);
    }

    /// End what `output` holds, whose calls completed: its pins, or its
    /// reserved pages, committed when every call was planned and succeeded
    /// and aborted otherwise.
    fn settle(&mut self, output: Output) {
        let Output {
            pool,
            keys,
            stage,
            failed,
            planned,
            ..
        } = output;
        match stage {
            Stage::Import(_) => self.unpin(&pool, keys),
            Stage::Export(slots) => {
                let reserved: Vec<Key> = keys
                    .into_iter()
                    .zip(&slots)
                    .filter(|(_, slot)| slot.is_some())
                    .map(|(key, _)| key)
                    .collect();
                if reserved.is_empty() {
                    return;
                }
                // Pages its calls wrote only in part are aborted.
                let request = if failed || !planned {
                    Request::Abort {
                        pool: pool.pool,
                        keys: reserved,
                    }
                } else {
                    Request::Commit {
                        pool: pool.pool,
                        keys: reserved,
                    }
                };
                self.servers.submit(pool.client, &request, Purpose::Settle);
            }
            Stage::Pinning | Stage::Reserving | Stage::Compute => {}
        }
    }

    /// End `request`'s lookup of `output`, which the scheduler no longer
    /// needs: its pins and its reservations end, also those not answered
    /// yet, and those of an output whose calls are in flight once they
    /// completed.
    pub(crate) fn drop_output(&mut self, request: RequestKey, output: OutputId) {
        let Some(outputs) = self.requests.get_mut(&request) else {
            return;
        };
        let Some(stored) = outputs.remove(&output) else {
            return;
        };
        if outputs.is_empty() {
            self.requests.remove(&request);
        }
        self.end(request, output, stored);
    }

    /// Forget admission `request`, which left the scheduler: the pins and
    /// the reservations of its outputs end, also those not answered yet,
    /// and those of an output whose calls are in flight once they completed.
    pub(crate) fn forget(&mut self, request: RequestKey) {
        let Some(outputs) = self.requests.remove(&request) else {
            return;
        };
        for (output, stored) in outputs {
            self.end(request, output, stored);
        }
    }

    /// End `request`'s lookup of `output`, removed from the store's
    /// lookups: an answer still awaited releases what it pinned or
    /// reserved, and an output whose calls are in flight ends once they
    /// completed, its reserved pages aborted.
    fn end(&mut self, request: RequestKey, output: OutputId, stored: Output) {
        match stored.stage {
            Stage::Pinning | Stage::Reserving => {
                for purpose in self.servers.pending_mut() {
                    match purpose {
                        Purpose::Pin {
                            request: owner,
                            output: owned,
                        } if *owner == request && *owned == output => {
                            *purpose = Purpose::Orphan {
                                pool: stored.pool.clone(),
                                keys: stored.keys.clone(),
                            };
                        }
                        Purpose::Reserve {
                            request: owner,
                            output: owned,
                        } if *owner == request && *owned == output => {
                            *purpose = Purpose::Abandoned {
                                pool: stored.pool.clone(),
                                keys: stored.keys.clone(),
                            };
                        }
                        _ => {}
                    }
                }
            }
            _ => {
                let stored = Output {
                    failed: true,
                    ..stored
                };
                if stored.calls > 0 {
                    self.left.insert((request, output), stored);
                } else {
                    self.settle(stored);
                }
            }
        }
    }

    /// `worker`'s ranks were lost, and with them the calls of theirs in
    /// flight: the outputs whose lookups ended end now, and those of
    /// admissions still running count no call in flight, so they end when
    /// their admission leaves.
    pub(crate) fn worker_failed(&mut self, worker: &WorkerGroupId) {
        for outputs in self.requests.values_mut() {
            for output in outputs.values_mut() {
                if output.pool.worker == *worker {
                    output.calls = 0;
                    output.failed = true;
                }
            }
        }
        // The lost calls never complete.
        let requests = &self.requests;
        let left = &self.left;
        self.calls.retain(|(request, _), output| {
            let stored = requests
                .get(request)
                .and_then(|outputs| outputs.get(output))
                .or_else(|| left.get(&(*request, *output)));
            stored.is_some_and(|stored| stored.pool.worker != *worker)
        });
        let lost: Vec<(RequestKey, OutputId)> = self
            .left
            .iter()
            .filter(|(_, output)| output.pool.worker == *worker)
            .map(|(key, _)| *key)
            .collect();
        for key in lost {
            if let Some(output) = self.left.remove(&key) {
                self.settle(output);
            }
        }
    }

    fn unpin(&mut self, pool: &GroupPool, keys: Vec<Key>) {
        if keys.is_empty() {
            return;
        }
        let unpin = Request::Unpin {
            pool: pool.pool,
            reader: pool.scope.clone(),
            keys,
        };
        self.servers.submit(pool.client, &unpin, Purpose::Settle);
    }

    fn abort(&mut self, pool: &GroupPool, keys: Vec<Key>) {
        if keys.is_empty() {
            return;
        }
        let abort = Request::Abort {
            pool: pool.pool,
            keys,
        };
        self.servers.submit(pool.client, &abort, Purpose::Settle);
    }

    /// An output's pins answered `slots`: every page pinned is imported;
    /// otherwise the pins end and every page is reserved, unless the
    /// request stores no output, which then computes it.
    fn pinned(&mut self, request: RequestKey, output: OutputId, slots: Vec<Option<u64>>) {
        let Some(stored) = self
            .requests
            .get_mut(&request)
            .and_then(|outputs| outputs.get_mut(&output))
        else {
            return;
        };
        if slots.len() == stored.keys.len() && slots.iter().all(Option::is_some) {
            tracing::info!(
                request = request.request_id.0,
                output = output.name,
                ordinal = output.ordinal,
                pages = slots.len(),
                "a request reads its encoder output from the host encoder pool"
            );
            stored.stage = Stage::Import(slots.into_iter().flatten().collect());
            return;
        }
        let pool = stored.pool.clone();
        let keys = stored.keys.clone();
        let held: Vec<Key> = keys
            .iter()
            .zip(&slots)
            .filter(|(_, slot)| slot.is_some())
            .map(|(key, _)| *key)
            .collect();
        let reserving = stored.write && {
            let reserve = Request::Reserve {
                pool: pool.pool,
                writer: pool.scope.clone(),
                keys,
                window: None,
            };
            self.servers
                .submit(pool.client, &reserve, Purpose::Reserve { request, output })
        };
        if let Some(stored) = self
            .requests
            .get_mut(&request)
            .and_then(|outputs| outputs.get_mut(&output))
        {
            stored.stage = if reserving {
                Stage::Reserving
            } else {
                Stage::Compute
            };
        }
        self.unpin(&pool, held);
    }

    /// An output's reservation answered `slots`: the calls write the pages
    /// reserved, unless a page could not be reserved, when they write none
    /// and the pages reserved are aborted.
    fn reserved(&mut self, request: RequestKey, output: OutputId, slots: Vec<PageSlot>) {
        let Some(stored) = self
            .requests
            .get_mut(&request)
            .and_then(|outputs| outputs.get_mut(&output))
        else {
            return;
        };
        let reserved: Vec<Option<u64>> = slots
            .iter()
            .map(|slot| match slot {
                PageSlot::Reserved { slot } => Some(*slot),
                PageSlot::Present | PageSlot::Full => None,
            })
            .collect();
        let complete = slots.len() == stored.keys.len()
            && !slots.iter().any(|slot| matches!(slot, PageSlot::Full));
        if complete && reserved.iter().any(Option::is_some) {
            tracing::info!(
                request = request.request_id.0,
                output = output.name,
                ordinal = output.ordinal,
                pages = reserved.iter().flatten().count(),
                "a request stores its encoder output in the host encoder pool"
            );
            stored.stage = Stage::Export(reserved);
            return;
        }
        tracing::debug!(
            request = request.request_id.0,
            output = output.name,
            ordinal = output.ordinal,
            ?slots,
            "a request computes its encoder output without storing it"
        );
        stored.stage = Stage::Compute;
        let pool = stored.pool.clone();
        let keys: Vec<Key> = stored
            .keys
            .iter()
            .zip(&reserved)
            .filter(|(_, slot)| slot.is_some())
            .map(|(key, _)| *key)
            .collect();
        self.abort(&pool, keys);
    }

    /// Take the servers' answers. Returns whether a lookup moved on.
    pub(crate) fn poll(&mut self) -> bool {
        let mut moved = false;
        for (purpose, reply) in self.servers.answers() {
            match (purpose, reply) {
                (Purpose::Pin { request, output }, reply) => {
                    moved = true;
                    let slots = match reply {
                        Reply::Pinned { slots } => slots,
                        other => {
                            tracing::warn!(?other, "a host encoder pool pin failed");
                            Vec::new()
                        }
                    };
                    self.pinned(request, output, slots);
                }
                (Purpose::Reserve { request, output }, reply) => {
                    moved = true;
                    let slots = match reply {
                        Reply::Reserved { slots } => slots,
                        other => {
                            tracing::warn!(?other, "a host encoder pool reservation failed");
                            Vec::new()
                        }
                    };
                    self.reserved(request, output, slots);
                }
                (Purpose::Orphan { pool, keys }, reply) => {
                    if let Reply::Pinned { slots } = reply {
                        let pinned = keys
                            .into_iter()
                            .zip(&slots)
                            .filter(|(_, slot)| slot.is_some())
                            .map(|(key, _)| key)
                            .collect();
                        self.unpin(&pool, pinned);
                    }
                }
                (Purpose::Abandoned { pool, keys }, reply) => {
                    if let Reply::Reserved { slots } = reply {
                        let reserved = keys
                            .into_iter()
                            .zip(&slots)
                            .filter(|(_, slot)| matches!(slot, PageSlot::Reserved { .. }))
                            .map(|(key, _)| key)
                            .collect();
                        self.abort(&pool, reserved);
                    }
                }
                (Purpose::Settle, Reply::Error { kind, message }) => {
                    tracing::warn!(?kind, message, "a host encoder pool request failed");
                }
                (Purpose::Settle, _) => {}
            }
        }
        moved
    }

    /// Whether a request to a server awaits its answer.
    pub(crate) fn waiting(&self) -> bool {
        self.servers.waiting()
    }
}

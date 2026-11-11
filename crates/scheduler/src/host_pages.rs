//! The scheduler's client of the deployment's host page pools: the
//! connections to the Storage Servers and the page lifecycle every store
//! drives in them.
//!
//! Each store that keeps derived state in the pools (the host KV cache and
//! KV storage, `kv_cache::offload`; latent trajectories, `trajectories`;
//! encoder outputs, `encoder_outputs`) states only its keys, which pages it
//! stores and when, and what each answer means to it (its own purpose `S`).
//! The page lifecycle is the same for all of them and is stated here once:
//! a reader pins the pages it reads ([`Servers::pin`]) and unpins them when
//! done ([`Servers::unpin`]); a writer reserves slots for the pages it
//! writes ([`Servers::reserve`]), copies into them, and commits them, or
//! aborts them when its copy failed ([`Servers::settle`]). A reader or a
//! writer that ends before its pins or reservation are answered turns them
//! into [`Purpose::Orphan`] or [`Purpose::Abandoned`] ([`Servers::pending_mut`]),
//! and the client releases whatever they took once they are answered, so
//! a store reads only the answers of its own purposes.
//!
//! The scheduler connects as a head of its engine's epoch, so what an epoch
//! on trial commits stays its own until its commit.
//!
//! **Peer reads.** The same lifecycle reaches the pools of other hosts
//! that serve the deployment (`kv_cache_config.peers`): the scheduler
//! connects to such a host's Storage Server as a [`Role::Peer`], pins the
//! published pages it lacks by key there ([`Servers::peer_pin`]), reserves
//! slots for them in its own host's pools with no writer
//! ([`Servers::reserve_read`]), and has its own host's Transfer Server READ
//! the peer's slots into them ([`Servers::read_pages`], spans whose source
//! is a [`Region::Remote`] page pool). Once the task ends it commits the
//! slots, or aborts them, and unpins at the peer ([`Servers::settle_read`]).
//! The pins live on the peer connection, so a scheduler that disappears
//! releases them with its connection. A peer endpoint is unauthenticated:
//! hosts that name each other as peers rely on the isolation of their
//! network. Requests to the servers
//! never block the scheduler: [`Servers::submit`] sends one with what it was
//! asked for (its purpose), and [`Servers::take`] takes the events and
//! answers that arrived, at every scheduler step.

use std::collections::HashMap;
use std::time::Duration;

use storage::protocol::{
    ErrorKind, Event, PageShape, PageSlot, PageWindow, Priority, Region, Reply, Request, Span,
};
use storage::{Client, Key, Role, Scope};

/// How long connecting to a peer host's Storage Server, or a blocking call
/// on that connection, may take before the peer is skipped.
const PEER_TIMEOUT: Duration = Duration::from_secs(2);

/// One page a peer read copies: the key, the peer pool's slot holding it
/// and the local slot it lands in: a slot of the local pool reserved for it,
/// or a slot of a staging tensor ([`PoolRead::staging`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PageRead {
    pub key: Key,
    pub source: u64,
    pub destination: u64,
}

/// The pages one transfer task reads from one peer pool into one local
/// pool, which [`Servers::settle_read`] commits or aborts and unpins once
/// the task ended; or, for a converted load, into slots of a rank's staging
/// tensor, which nothing commits.
#[derive(Debug, Clone)]
pub(crate) struct PoolRead {
    /// The connection to the local pool's server, which submits the task.
    pub client: usize,
    pub pool: u64,
    /// The connection to the peer's server, which holds the pins.
    pub peer: usize,
    /// The peer's endpoint, which the task's sources name.
    pub endpoint: String,
    pub peer_pool: u64,
    /// The shape of the peer pool's pages, which the destination slots have.
    pub shape: PageShape,
    pub pages: Vec<PageRead>,
    /// The staging tensor the pages land in, in place of `pool`: an exclusive
    /// host tensor of a rank, retained at the local server, whose slots hold
    /// pages of the peer's shape (`ipc::PoolConversion`).
    pub staging: Option<Key>,
}

impl PoolRead {
    /// The task's spans: each page whole, from its peer slot to its local
    /// slot. The destination slots have the peer pool's page shape (layout
    /// and head slice), so a page is one span.
    fn spans(&self) -> Vec<Span> {
        let bytes = self.shape.page_bytes;
        let destination = match self.staging {
            Some(key) => Region::Tensor { key, at: None },
            None => Region::Pool { pool: self.pool },
        };
        self.pages
            .iter()
            .map(|page| Span {
                source: Region::Remote {
                    endpoint: self.endpoint.clone(),
                    region: Box::new(Region::Pool {
                        pool: self.peer_pool,
                    }),
                },
                source_offset: page.source * bytes,
                destination: destination.clone(),
                destination_offset: page.destination * bytes,
                bytes,
                rows: 1,
                source_pitch: 0,
                destination_pitch: 0,
            })
            .collect()
    }

    pub(crate) fn keys(&self) -> Vec<Key> {
        self.pages.iter().map(|page| page.key).collect()
    }

    /// The bytes the read copies.
    pub(crate) fn bytes(&self) -> u64 {
        self.shape.page_bytes * self.pages.len() as u64
    }
}

/// What [`Servers::take`] takes: the events, each with the connection it
/// arrived on, and the answers of the store's own purposes.
pub(crate) type Arrivals<S> = (Vec<(usize, Event)>, Vec<(S, Reply)>);

/// What a request to the host page pools is for: one of the store's own
/// (`S`), whose answer the store reads, or an end of the page lifecycle,
/// whose answer the client settles itself ([`Servers::take`]).
pub(crate) enum Purpose<S> {
    /// A request whose answer the store reads.
    Store(S),
    /// Pins of `keys` in `pool` for `reader`, answered after the reader
    /// ended: the pages they pinned are unpinned. A pool counts pins per
    /// reader, so only the pages the answer names as pinned are.
    Orphan {
        client: usize,
        pool: u64,
        reader: Scope,
        keys: Vec<Key>,
    },
    /// Pins of `keys` at the peer of connection `peer`, answered after their
    /// reader ended: every page they pinned is unpinned at the peer.
    PeerOrphan { peer: usize, keys: Vec<Key> },
    /// A reservation of `keys` in `pool`, answered after its writer ended:
    /// the slots it reserved are aborted.
    Abandoned {
        client: usize,
        pool: u64,
        keys: Vec<Key>,
    },
    /// A request whose answer only reports a failure: an unpin, a commit,
    /// an abort, a cancel or a prefetch.
    Settle,
}

/// The keys of `keys` whose pages a pin's answer `slots` pinned.
fn pinned(keys: &[Key], slots: &[Option<u64>]) -> Vec<Key> {
    keys.iter()
        .zip(slots)
        .filter(|(_, slot)| slot.is_some())
        .map(|(key, _)| *key)
        .collect()
}

/// The connections to the servers of the pools one store uses, with the
/// purpose of every request they have not answered.
pub(crate) struct Servers<S> {
    deployment: String,
    epoch: u64,
    /// What the store keeps, for its logs.
    what: &'static str,
    /// The engine's host's server first, then other hosts' by the TCP
    /// address they were reached at; `None` once a peer connection failed.
    clients: Vec<Option<Client>>,
    remote: HashMap<String, usize>,
    /// The connections to peer hosts' servers, as a [`Role::Peer`], by
    /// endpoint.
    peers: HashMap<String, usize>,
    /// Requests submitted and not yet answered, by connection and request.
    pending: HashMap<(usize, u64), Purpose<S>>,
}

impl<S> Servers<S> {
    /// Connect to this host's Storage Server, at `socket`, as a head of
    /// `deployment` in `epoch`, for a store keeping `what`.
    pub(crate) fn connect(
        socket: &std::path::Path,
        deployment: &str,
        epoch: u64,
        what: &'static str,
    ) -> anyhow::Result<Self> {
        let client = Client::connect(
            socket,
            Role::Head {
                deployment: deployment.to_owned(),
                epoch,
            },
        )?;
        Ok(Self {
            deployment: deployment.to_owned(),
            epoch,
            what,
            clients: vec![Some(client)],
            remote: HashMap::new(),
            peers: HashMap::new(),
            pending: HashMap::new(),
        })
    }

    /// The deployment the store serves.
    pub(crate) fn deployment(&self) -> &str {
        &self.deployment
    }

    /// The connection to the server of a rank's pool: the engine's host's
    /// for a `local` rank, otherwise the one reached at the rank's
    /// `endpoint`, connected when first named. `None` when a remote rank
    /// names no endpoint or its server cannot be reached.
    pub(crate) fn client(&mut self, local: bool, endpoint: Option<&str>) -> Option<usize> {
        if local {
            return Some(0);
        }
        let address = endpoint?;
        if let Some(&client) = self.remote.get(address) {
            return Some(client);
        }
        let role = Role::Head {
            deployment: self.deployment.clone(),
            epoch: self.epoch,
        };
        match Client::connect_tcp(address, role) {
            Ok(client) => {
                self.clients.push(Some(client));
                self.remote
                    .insert(address.to_owned(), self.clients.len() - 1);
                Some(self.clients.len() - 1)
            }
            Err(error) => {
                tracing::warn!(address, %error, what = self.what, "a storage server cannot be reached");
                None
            }
        }
    }

    /// Send `request` on connection `client` and wait for its answer: for
    /// the requests that size the pools before the store serves, never on
    /// a scheduler step.
    ///
    /// # Errors
    ///
    /// When the server cannot be reached or the request fails.
    pub(crate) fn call(&mut self, client: usize, request: &Request) -> anyhow::Result<Reply> {
        let connection = self.clients[client]
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("the storage server connection failed"))?;
        Ok(connection.call(request, &[])?.reply)
    }

    /// The connection to the Storage Server of the peer host at `endpoint`
    /// as a [`Role::Peer`], connected when first named or after it failed.
    /// `None` when the peer cannot be reached now; the next admission tries
    /// again, without a retry loop.
    pub(crate) fn peer(&mut self, endpoint: &str) -> Option<usize> {
        if let Some(&client) = self.peers.get(endpoint)
            && self.clients[client].is_some()
        {
            return Some(client);
        }
        match Client::connect_peer(endpoint, PEER_TIMEOUT) {
            Ok(connection) => {
                let client = match self.peers.get(endpoint) {
                    Some(&client) => {
                        self.clients[client] = Some(connection);
                        client
                    }
                    None => {
                        self.clients.push(Some(connection));
                        self.clients.len() - 1
                    }
                };
                self.peers.insert(endpoint.to_owned(), client);
                Some(client)
            }
            Err(error) => {
                tracing::warn!(endpoint, %error, what = self.what, "a peer storage server cannot be reached");
                None
            }
        }
    }

    /// Pin the published pages of `keys` of the deployment at the peer of
    /// connection `peer`, in its pools of the page `layouts`, answered
    /// (`Reply::PeerPinned`) for `purpose`.
    pub(crate) fn peer_pin(
        &mut self,
        peer: usize,
        layouts: Vec<Key>,
        keys: Vec<Key>,
        purpose: S,
    ) -> bool {
        let request = Request::PeerPin {
            deployment: self.deployment.clone(),
            layouts,
            keys,
        };
        self.submit(peer, &request, Purpose::Store(purpose))
    }

    /// End the pins of `keys` in the peer's `pool` this connection holds;
    /// nothing for no key.
    pub(crate) fn peer_unpin(&mut self, peer: usize, pool: u64, keys: Vec<Key>) -> bool {
        if keys.is_empty() {
            return true;
        }
        self.submit(peer, &Request::PeerUnpin { pool, keys }, Purpose::Settle)
    }

    /// Reserve slots for `keys` in `pool` for the head's own transfer task,
    /// a peer read, answered (`Reply::Reserved`) for `purpose`.
    pub(crate) fn reserve_read(
        &mut self,
        client: usize,
        pool: u64,
        keys: Vec<Key>,
        purpose: S,
    ) -> bool {
        let reserve = Request::Reserve {
            pool,
            writer: None,
            keys,
            window: None,
        };
        self.submit(client, &reserve, Purpose::Store(purpose))
    }

    /// Submit the transfer task of `read` to its local server, at the
    /// priority of what a request waits for, answered (`Reply::Transferring`)
    /// for `purpose`; its end arrives as an [`Event::Transferred`] on that
    /// connection.
    pub(crate) fn read_pages(&mut self, read: &PoolRead, purpose: S) -> bool {
        let request = Request::Transfer {
            spans: read.spans(),
            priority: Priority::Request,
            update: 0,
        };
        self.submit(read.client, &request, Purpose::Store(purpose))
    }

    /// Cancel the transfer task `task` of connection `client`; its
    /// [`Event::Transferred`] follows once it copies nothing any more.
    pub(crate) fn cancel_read(&mut self, client: usize, task: u64) -> bool {
        let cancel = Request::Cancel {
            tasks: vec![task],
            update: None,
        };
        self.submit(client, &cancel, Purpose::Settle)
    }

    /// Settle `read` once its task ended: commit the local slots when every
    /// page landed (`complete`), abort them otherwise, and end the pins at
    /// the peer either way. A read into a staging tensor reserved no pool
    /// slot, so only its pins end.
    pub(crate) fn settle_read(&mut self, read: &PoolRead, complete: bool) {
        let keys = read.keys();
        if read.staging.is_none() {
            self.settle(read.client, read.pool, keys.clone(), complete);
        }
        self.peer_unpin(read.peer, read.peer_pool, keys);
    }

    /// Submit `request` on connection `client`, answered for `purpose`.
    /// Returns whether it was sent.
    pub(crate) fn submit(&mut self, client: usize, request: &Request, purpose: Purpose<S>) -> bool {
        let Some(connection) = self.clients[client].as_mut() else {
            return false;
        };
        match connection.submit(request, &[]) {
            Ok(id) => {
                self.pending.insert((client, id), purpose);
                true
            }
            Err(error) => {
                tracing::warn!(%error, what = self.what, "a storage server request failed");
                false
            }
        }
    }

    /// Pin `keys` in `pool` on connection `client` for `reader`, answered
    /// (`Reply::Pinned`) for `purpose`: the slots of the pages the pool
    /// holds, which it keeps until the reader unpins them.
    pub(crate) fn pin(
        &mut self,
        client: usize,
        pool: u64,
        reader: &Scope,
        keys: Vec<Key>,
        purpose: S,
    ) -> bool {
        let pin = Request::Pin {
            pool,
            reader: reader.clone(),
            keys,
        };
        self.submit(client, &pin, Purpose::Store(purpose))
    }

    /// End `reader`'s pins of `keys` in `pool`; nothing for no key.
    pub(crate) fn unpin(
        &mut self,
        client: usize,
        pool: u64,
        reader: &Scope,
        keys: Vec<Key>,
    ) -> bool {
        if keys.is_empty() {
            return true;
        }
        let unpin = Request::Unpin {
            pool,
            reader: reader.clone(),
            keys,
        };
        self.submit(client, &unpin, Purpose::Settle)
    }

    /// Reserve slots for `keys` in `pool` for `writer`, the pages of one
    /// `window` evicted whole when one is named, answered
    /// (`Reply::Reserved`) for `purpose`.
    pub(crate) fn reserve(
        &mut self,
        client: usize,
        pool: u64,
        writer: &Scope,
        keys: Vec<Key>,
        window: Option<PageWindow>,
        purpose: S,
    ) -> bool {
        let reserve = Request::Reserve {
            pool,
            writer: Some(writer.clone()),
            keys,
            window,
        };
        self.submit(client, &reserve, Purpose::Store(purpose))
    }

    /// Commit the reserved `keys` in `pool` once their copies landed, or
    /// abort them when `written` is false; nothing for no key.
    pub(crate) fn settle(
        &mut self,
        client: usize,
        pool: u64,
        keys: Vec<Key>,
        written: bool,
    ) -> bool {
        if keys.is_empty() {
            return true;
        }
        let request = if written {
            Request::Commit { pool, keys }
        } else {
            Request::Abort { pool, keys }
        };
        self.submit(client, &request, Purpose::Settle)
    }

    /// The purposes of the requests not yet answered.
    pub(crate) fn pending(&self) -> impl Iterator<Item = &Purpose<S>> {
        self.pending.values()
    }

    /// The purposes of the requests not yet answered, to be changed when
    /// what asked for them ended: a store's pins become an
    /// [`Purpose::Orphan`], its reservation an [`Purpose::Abandoned`].
    pub(crate) fn pending_mut(&mut self) -> impl Iterator<Item = &mut Purpose<S>> {
        self.pending.values_mut()
    }

    /// Settles the answer of a lifecycle end: releases what an orphaned pin
    /// or an abandoned reservation took, and reports a failure.
    fn settle_answer(&mut self, purpose: Purpose<S>, reply: Reply) {
        match (purpose, reply) {
            (Purpose::Store(_), _) => {}
            (
                Purpose::Orphan {
                    client,
                    pool,
                    reader,
                    keys,
                },
                Reply::Pinned { slots },
            ) => {
                self.unpin(client, pool, &reader, pinned(&keys, &slots));
            }
            (Purpose::PeerOrphan { peer, keys }, Reply::PeerPinned { pools }) => {
                for pool in pools {
                    self.peer_unpin(peer, pool.pool, pinned(&keys, &pool.slots));
                }
            }
            (Purpose::Abandoned { client, pool, keys }, Reply::Reserved { slots }) => {
                let reserved = keys
                    .into_iter()
                    .zip(&slots)
                    .filter(|(_, slot)| matches!(slot, PageSlot::Reserved { .. }))
                    .map(|(key, _)| key)
                    .collect();
                self.settle(client, pool, reserved, false);
            }
            (_, Reply::Error { kind, message }) => {
                tracing::warn!(
                    ?kind,
                    message,
                    what = self.what,
                    "a host page pool request failed"
                );
            }
            _ => {}
        }
    }

    /// The events, each with the connection it arrived on, and the answers
    /// of the store's own purposes that arrived since the last call, each
    /// with its purpose; the answers of lifecycle ends are settled here.
    ///
    /// A peer connection that failed is dropped, and each request it will
    /// not answer is answered with [`Reply::Error`]; the connections to the
    /// deployment's own hosts stay, as their failures are their ranks'.
    pub(crate) fn take(&mut self) -> Arrivals<S> {
        let mut events = Vec::new();
        let mut failed = Vec::new();
        for (index, client) in self.clients.iter_mut().enumerate() {
            let Some(connection) = client.as_mut() else {
                continue;
            };
            if let Err(error) = connection.pump() {
                tracing::warn!(%error, what = self.what, "a storage server connection failed");
                failed.push(index);
            }
            while let Some((event, _)) = connection.next_event() {
                events.push((index, event));
            }
        }
        let mut answers = Vec::new();
        for client in failed {
            if !self.peers.values().any(|&peer| peer == client) {
                continue;
            }
            self.clients[client] = None;
            let lost: Vec<(usize, u64)> = self
                .pending
                .keys()
                .copied()
                .filter(|(connection, _)| *connection == client)
                .collect();
            for id in lost {
                if let Some(purpose) = self.pending.remove(&id) {
                    let reply = Reply::Error {
                        kind: ErrorKind::Unavailable,
                        message: "the peer storage server connection failed".to_owned(),
                    };
                    answers.push((purpose, reply));
                }
            }
        }
        let ids: Vec<(usize, u64)> = self.pending.keys().copied().collect();
        for id in ids {
            let Some(answer) = self.clients[id.0]
                .as_mut()
                .and_then(|client| client.take(id.1))
            else {
                continue;
            };
            if let Some(purpose) = self.pending.remove(&id) {
                answers.push((purpose, answer.reply));
            }
        }
        let mut own = Vec::with_capacity(answers.len());
        for (purpose, reply) in answers {
            match purpose {
                Purpose::Store(purpose) => own.push((purpose, reply)),
                lifecycle => self.settle_answer(lifecycle, reply),
            }
        }
        (events, own)
    }

    /// The answers that arrived since the last call, each with its purpose,
    /// for a store that awaits no event.
    pub(crate) fn answers(&mut self) -> Vec<(S, Reply)> {
        self.take().1
    }

    /// Whether a request awaits its answer.
    #[cfg(test)]
    pub(crate) fn waiting(&self) -> bool {
        !self.pending.is_empty()
    }

    /// The connections' descriptors, each readable when an answer or an
    /// event arrived on it: what the scheduler waits on beside its wake.
    pub(crate) fn descriptors(&self) -> impl Iterator<Item = std::os::fd::BorrowedFd<'_>> {
        self.clients.iter().flatten().map(storage::Client::as_fd)
    }
}

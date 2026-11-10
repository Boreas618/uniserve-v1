//! The scheduler's client of the deployment's host page pools: the
//! connections to the Storage Servers and the page lifecycle every store
//! drives in them.
//!
//! Each store that keeps derived state in the pools (the host KV cache and
//! KV storage, `kv_cache::offload`; latent trajectories, `trajectories`;
//! encoder outputs, `encoder_outputs`) states only its keys, which pages it
//! stores and when, and what each answer means to it (its purpose `P`).
//! The page lifecycle is the same for all of them and is stated here once:
//! a reader pins the pages it reads ([`Servers::pin`]) and unpins them when
//! done ([`Servers::unpin`]); a writer reserves slots for the pages it
//! writes ([`Servers::reserve`]), copies into them, and commits them, or
//! aborts them when its copy failed ([`Servers::settle`]).
//!
//! The scheduler connects as a head of its engine's epoch, so what an epoch
//! on trial commits stays its own until its commit. Requests to the servers
//! never block the scheduler: [`Servers::submit`] sends one with what it was
//! asked for (its purpose), and [`Servers::take`] takes the events and
//! answers that arrived, at every scheduler step.

use std::collections::HashMap;

use storage::protocol::{Event, PageWindow, Reply, Request};
use storage::{Client, Key, Role, Scope};

/// The connections to the servers of the pools one store uses, with the
/// purpose of every request they have not answered.
pub(crate) struct Servers<P> {
    deployment: String,
    epoch: u64,
    /// What the store keeps, for its logs.
    what: &'static str,
    /// The engine's host's server first, then other hosts' by the TCP
    /// address they were reached at.
    clients: Vec<Client>,
    remote: HashMap<String, usize>,
    /// Requests submitted and not yet answered, by connection and request.
    pending: HashMap<(usize, u64), P>,
}

impl<P> Servers<P> {
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
            clients: vec![client],
            remote: HashMap::new(),
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
                self.clients.push(client);
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
        Ok(self.clients[client].call(request, &[])?.reply)
    }

    /// Submit `request` on connection `client`, answered for `purpose`.
    /// Returns whether it was sent.
    pub(crate) fn submit(&mut self, client: usize, request: &Request, purpose: P) -> bool {
        match self.clients[client].submit(request, &[]) {
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
        purpose: P,
    ) -> bool {
        let pin = Request::Pin {
            pool,
            reader: reader.clone(),
            keys,
        };
        self.submit(client, &pin, purpose)
    }

    /// End `reader`'s pins of `keys` in `pool`; nothing for no key.
    pub(crate) fn unpin(
        &mut self,
        client: usize,
        pool: u64,
        reader: &Scope,
        keys: Vec<Key>,
        purpose: P,
    ) -> bool {
        if keys.is_empty() {
            return true;
        }
        let unpin = Request::Unpin {
            pool,
            reader: reader.clone(),
            keys,
        };
        self.submit(client, &unpin, purpose)
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
        purpose: P,
    ) -> bool {
        let reserve = Request::Reserve {
            pool,
            writer: writer.clone(),
            keys,
            window,
        };
        self.submit(client, &reserve, purpose)
    }

    /// Commit the reserved `keys` in `pool` once their copies landed, or
    /// abort them when `written` is false; nothing for no key.
    pub(crate) fn settle(
        &mut self,
        client: usize,
        pool: u64,
        keys: Vec<Key>,
        written: bool,
        purpose: P,
    ) -> bool {
        if keys.is_empty() {
            return true;
        }
        let request = if written {
            Request::Commit { pool, keys }
        } else {
            Request::Abort { pool, keys }
        };
        self.submit(client, &request, purpose)
    }

    /// The purposes of the requests not yet answered.
    pub(crate) fn pending(&self) -> impl Iterator<Item = &P> {
        self.pending.values()
    }

    /// The purposes of the requests not yet answered, to be changed when
    /// what asked for them ended.
    pub(crate) fn pending_mut(&mut self) -> impl Iterator<Item = &mut P> {
        self.pending.values_mut()
    }

    /// The events and the answers that arrived since the last call, each
    /// answer with its purpose.
    pub(crate) fn take(&mut self) -> (Vec<Event>, Vec<(P, Reply)>) {
        let mut events = Vec::new();
        for client in &mut self.clients {
            if let Err(error) = client.pump() {
                tracing::warn!(%error, what = self.what, "a storage server connection failed");
            }
            while let Some((event, _)) = client.next_event() {
                events.push(event);
            }
        }
        let ids: Vec<(usize, u64)> = self.pending.keys().copied().collect();
        let mut answers = Vec::new();
        for id in ids {
            let Some(answer) = self.clients[id.0].take(id.1) else {
                continue;
            };
            if let Some(purpose) = self.pending.remove(&id) {
                answers.push((purpose, answer.reply));
            }
        }
        (events, answers)
    }

    /// The answers that arrived since the last call, each with its purpose,
    /// for a store that awaits no event.
    pub(crate) fn answers(&mut self) -> Vec<(P, Reply)> {
        self.take().1
    }

    /// Whether a request awaits its answer.
    pub(crate) fn waiting(&self) -> bool {
        !self.pending.is_empty()
    }
}

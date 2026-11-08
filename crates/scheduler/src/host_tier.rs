//! A scheduler's connections to the deployment's Storage Servers, through
//! which it keeps derived state in their host page pools (trajectories,
//! encoder outputs).
//!
//! The scheduler connects as a head of its engine's world, so what a world
//! on trial commits stays its own until its commit. Requests to the servers
//! never block the scheduler: [`Servers::submit`] sends one with what it was
//! asked for (its purpose), and [`Servers::answers`] takes the answers that
//! arrived, at every scheduler step.

use std::collections::HashMap;

use storage::protocol::{Reply, Request};
use storage::{Client, Role};

/// The connections to the servers of the pools one store uses, with the
/// purpose of every request they have not answered.
pub(crate) struct Servers<P> {
    deployment: String,
    world: u64,
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
    /// `deployment` in `world`, for a store keeping `what`.
    pub(crate) fn connect(
        socket: &std::path::Path,
        deployment: &str,
        world: u64,
        what: &'static str,
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
                tracing::warn!(address, %error, what = self.what, "a storage server cannot be reached");
                None
            }
        }
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

    /// The purposes of the requests not yet answered.
    pub(crate) fn pending(&self) -> impl Iterator<Item = &P> {
        self.pending.values()
    }

    /// The purposes of the requests not yet answered, to be changed when
    /// what asked for them ended.
    pub(crate) fn pending_mut(&mut self) -> impl Iterator<Item = &mut P> {
        self.pending.values_mut()
    }

    /// The answers that arrived since the last call, each with its purpose.
    pub(crate) fn answers(&mut self) -> Vec<(P, Reply)> {
        for client in &mut self.clients {
            if let Err(error) = client.pump() {
                tracing::warn!(%error, what = self.what, "a storage server connection failed");
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
        answers
    }

    /// Whether a request awaits its answer.
    pub(crate) fn waiting(&self) -> bool {
        !self.pending.is_empty()
    }
}

//! The server's side of the KV storage (L3), which its Mooncake connector
//! serves.
//!
//! The server runs the connector ([`crate::connector`]) as a child process
//! and starts it again when it ends. The connector connects as
//! [`Role::Connector`](crate::protocol::Role::Connector); the server sends it
//! every page pool and its tasks as events, and the connector reports each
//! task's results with [`Request::Finished`](crate::protocol::Request).
//!
//! While no connector is connected, L3 holds nothing: a lookup counts L2
//! only, a prefetch fetches nothing, and committed pages are not uploaded.
//! A connector that ends fails its tasks in flight.

use std::collections::{BTreeSet, HashMap};

use storage::protocol::Key;

use crate::child::{Launch, Supervised};

/// A task the connector runs for the server.
pub(super) enum Task {
    /// The L3 part of a lookup: `keys` are the lookup's keys the pool did
    /// not hold, and `held` says for each key of the lookup whether the pool
    /// held it. The lookup's client waits for it.
    Lookup {
        client: u64,
        request: u64,
        pool: u64,
        keys: Vec<Key>,
        held: Vec<bool>,
    },
    /// Committed pages being written to Mooncake from their slots, which
    /// stay out of eviction until the task finishes.
    Upload { pool: u64, pages: Vec<(Key, u32)> },
    /// Pages being read from Mooncake into slots reserved for them, for the
    /// prefetch `tag` of `head`, which named `keys`.
    Download {
        head: u64,
        tag: u64,
        pool: u64,
        pages: Vec<(Key, u32)>,
        keys: Vec<Key>,
    },
}

/// The connector process and its connection.
pub(super) struct KvStorage {
    pub(super) process: Supervised,
    /// The connected connector's client.
    pub(super) client: Option<u64>,
    pub(super) tasks: HashMap<u64, Task>,
    next: u64,
    /// The pools the connected connector was given.
    pub(super) offered: BTreeSet<u64>,
}

impl KvStorage {
    pub(super) fn new(launch: Option<Launch>) -> Self {
        Self {
            process: Supervised::new("Mooncake connector", launch),
            client: None,
            tasks: HashMap::new(),
            next: 1,
            offered: BTreeSet::new(),
        }
    }

    /// The identity of a new task.
    pub(super) fn next_task(&mut self) -> u64 {
        let id = self.next;
        self.next += 1;
        id
    }
}

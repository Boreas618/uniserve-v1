//! The server's side of the KV storage (L3), which its Mooncake connector
//! serves.
//!
//! The server runs the connector ([`crate::mooncake`]) as a child process
//! and starts it again when it ends. The connector connects as
//! [`Role::Connector`](::storage::protocol::Role::Connector); the server
//! sends it every part of a buffer pool's host region as a pool of pages
//! and its tasks as events, and the connector reports each task's results
//! with [`Request::Finished`](::storage::protocol::Request::Finished).
//!
//! While no connector is connected, L3 holds nothing: a task fails at once.
//! A connector that ends loses its tasks in flight.

use std::collections::{BTreeSet, HashMap};

use ::storage::protocol::Key;

use crate::child::{Launch, Process};

/// A task the connector runs for the server: the pages of a buffer pool
/// task (the server's number `number`) in one part of a host region, the
/// connector's pool `pool`, written to the KV storage when `upload` and read
/// from it otherwise. `keys` names, for each page, the copy of the task it
/// is part of.
pub(super) struct Task {
    pub(super) number: u64,
    pub(super) pool: u64,
    pub(super) pages: Vec<(Key, u32)>,
    pub(super) keys: Vec<usize>,
    pub(super) upload: bool,
}

/// The connector process and its connection.
pub(super) struct KvStorage {
    pub(super) process: Process,
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
            process: Process::new("Mooncake connector", launch),
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

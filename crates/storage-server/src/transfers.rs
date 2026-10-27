//! The server's side of its Transfer Server, which copies bytes between the
//! memory the server holds.
//!
//! The server runs the Transfer Server ([`crate::transfer`]) as a child
//! process ([`Supervised`]) and starts it again when it ends. It connects
//! as [`Role::Transfer`](storage::protocol::Role::Transfer), reports the
//! bandwidth of every link and its CUDA contexts' device memory
//! ([`Request::Measured`](storage::protocol::Request::Measured)), and runs
//! the tasks heads and workers submit:
//!
//! 1. A client's [`Request::Transfer`] names its spans by [`Region`]. The
//!    server checks that every region is memory of the client's deployment
//!    and every span lies within its region and writes no read-only tensor,
//!    resolves each region to an [`Allocation`] and a byte offset, and
//!    forwards the task as [`Event::Transfer`]. The first task that copies
//!    an allocation gives the Transfer Server its descriptor
//!    ([`Event::Allocation`]); a released allocation is taken back
//!    ([`Event::AllocationReleased`]) and unmapped once no task copies it.
//! 2. The Transfer Server reports each task's progress and end
//!    ([`Request::Copied`]), which the server relays to the submitter
//!    ([`Event::TransferProgress`], [`Event::Transferred`]).
//! 3. A cancellation, by task or by update, and a submitter whose connection
//!    closes cancel tasks ([`Event::Cancel`]); each still ends with its
//!    report, since bytes may be in flight until then.
//!
//! When the Transfer Server's connection closes, every task it held fails
//! and the submitters learn it at once: the update a task served aborts,
//! and the next Transfer Server maps the allocations of new tasks again.
//! While none is connected a submission is refused
//! ([`ErrorKind::Unavailable`]).

use std::collections::{BTreeMap, BTreeSet};

use storage::protocol::{
    Access, Allocation, ContextBytes, ErrorKind, Event, Link, LinkBandwidth, ListedTask,
    ListedTransfer, Memory, Move, Placement, Priority, Region, Reply, Role, Span, Throttle,
    TransferEnd,
};

use crate::child::{Launch, Supervised};
use crate::{Attachment, ClientId, Outcome, Server, StorageId, invalid};

/// A task the Transfer Server runs, as the server keeps it.
pub(super) struct Task {
    /// The submitter, which the task's events go to.
    client: ClientId,
    deployment: String,
    update: u64,
    priority: Priority,
    /// The bytes of all its spans, and those copied so far.
    bytes: u64,
    copied: u64,
    cancelled: bool,
}

/// The Transfer Server process, its connection and its tasks.
pub(super) struct Transfers {
    pub(super) process: Supervised,
    /// The connected Transfer Server's client.
    pub(super) client: Option<ClientId>,
    pub(super) tasks: BTreeMap<u64, Task>,
    next_task: u64,
    /// The allocations the connected Transfer Server was given.
    given: BTreeSet<Allocation>,
    /// What it measured when it started.
    links: Vec<LinkBandwidth>,
    contexts: Vec<ContextBytes>,
}

impl Transfers {
    pub(super) fn new(launch: Option<Launch>) -> Self {
        Self {
            process: Supervised::new("Transfer Server", launch),
            client: None,
            tasks: BTreeMap::new(),
            next_task: 1,
            given: BTreeSet::new(),
            links: Vec::new(),
            contexts: Vec::new(),
        }
    }
}

/// A region resolved to one of the server's allocations: the allocation,
/// the memory it lies in, the offset of the region's first byte and the
/// allocation's size.
struct Resolved {
    allocation: Allocation,
    memory: Memory,
    base: u64,
    bytes: u64,
    /// The descriptor the Transfer Server needs, unless it is an
    /// allocation of another host.
    attachment: Option<Attachment>,
    /// Whether the Transfer Server maps it for writing.
    writable: bool,
}

impl Server {
    /// The deployment of a head's or a worker's role.
    fn deployment_of(role: &Role) -> Option<&str> {
        match role {
            Role::Head { deployment, .. } => Some(deployment),
            Role::Worker { scope, .. } => Some(&scope.deployment),
            _ => None,
        }
    }

    /// Whether the client `id` may connect as the Transfer Server: it is the
    /// server's running Transfer Server process, and none is connected.
    pub(super) fn transfer_hello(&mut self, id: ClientId, pid: i32) -> Result<(), String> {
        if self.transfers.client.is_some() {
            return Err("the server has a Transfer Server".into());
        }
        // The Transfer Server receives the descriptor of every allocation
        // it copies, so only the process the server started may be one.
        let child = self.transfers.process.pid();
        if child.is_none_or(|child| i64::from(child) != i64::from(pid)) {
            return Err("only the server's own Transfer Server connects as one".into());
        }
        self.transfers.client = Some(id);
        tracing::info!(pid, "the Transfer Server connected");
        Ok(())
    }

    /// A head's or a worker's transfer request.
    pub(super) fn transfer_request(
        &mut self,
        id: ClientId,
        request: &storage::protocol::Request,
        role: &Role,
    ) -> Result<Outcome, (ErrorKind, String)> {
        use storage::protocol::Request;

        let deployment = Self::deployment_of(role)
            .ok_or_else(|| invalid("a transfer request comes from a head or a worker"))?
            .to_owned();
        match request {
            Request::Transfer {
                spans,
                priority,
                update,
            } => {
                let task = self.submit(id, &deployment, spans, *priority, *update)?;
                Ok(Outcome::Reply(Reply::Transferring { task }, Vec::new()))
            }
            Request::Cancel { tasks, update } => {
                let selected: BTreeSet<u64> = tasks.iter().copied().collect();
                let cancelled = self.cancel_where(|task_id, task| {
                    task.deployment == deployment
                        && (selected.contains(&task_id) || Some(task.update) == *update)
                });
                Ok(Outcome::Reply(
                    Reply::Cancelled { tasks: cancelled },
                    Vec::new(),
                ))
            }
            Request::Throttle { links, throttle } => {
                if !matches!(role, Role::Head { .. }) {
                    return Err(invalid("a head throttles links"));
                }
                self.throttle(links, *throttle)?;
                Ok(Outcome::Reply(Reply::Throttled, Vec::new()))
            }
            _ => Err(invalid("not a transfer request")),
        }
    }

    /// A report of the Transfer Server's.
    pub(super) fn transfer_report(
        &mut self,
        request: &storage::protocol::Request,
    ) -> Result<Outcome, (ErrorKind, String)> {
        use storage::protocol::Request;

        match request {
            Request::Measured { links, contexts } => {
                self.transfers.links.clone_from(links);
                self.transfers.contexts.clone_from(contexts);
                for context in contexts {
                    tracing::info!(
                        gpu = %cuda::DeviceUuid(context.gpu),
                        bytes = context.bytes,
                        "the Transfer Server's CUDA context"
                    );
                }
                Ok(Outcome::Reply(Reply::Noted, Vec::new()))
            }
            Request::Copied { task, bytes, end } => {
                self.copied(*task, *bytes, end.clone());
                Ok(Outcome::Reply(Reply::Noted, Vec::new()))
            }
            _ => Err(invalid("not a request of the Transfer Server")),
        }
    }

    /// Check and forward a transfer; returns its task.
    fn submit(
        &mut self,
        client: ClientId,
        deployment: &str,
        spans: &[Span],
        priority: Priority,
        update: u64,
    ) -> Result<u64, (ErrorKind, String)> {
        let Some(transfer_server) = self.transfers.client else {
            return Err(unavailable());
        };
        if spans.is_empty() {
            return Err(invalid("a transfer copies at least one span"));
        }

        // Every span is checked before anything is given or forwarded, so
        // a refused task copies nothing.
        let mut moves = Vec::with_capacity(spans.len());
        let mut needed: BTreeMap<Allocation, Resolved> = BTreeMap::new();
        let mut bytes = 0_u64;
        for span in spans {
            if span.bytes == 0 {
                return Err(invalid("a span copies at least one byte"));
            }
            let source = self.resolve(deployment, &span.source, false)?;
            let destination = self.resolve(deployment, &span.destination, true)?;
            if matches!(source.allocation, Allocation::Fabric { .. })
                && matches!(destination.allocation, Allocation::Fabric { .. })
            {
                return Err(invalid(
                    "a span copies to or from this host's memory, not between two other hosts",
                ));
            }
            let source_offset = within(&source, span.source_offset, span.bytes, "source")?;
            let destination_offset = within(
                &destination,
                span.destination_offset,
                span.bytes,
                "destination",
            )?;
            bytes = bytes
                .checked_add(span.bytes)
                .ok_or_else(|| invalid("the spans' bytes overflow"))?;
            moves.push(Move {
                source: source.allocation.clone(),
                source_offset,
                destination: destination.allocation.clone(),
                destination_offset,
                bytes: span.bytes,
            });
            for resolved in [source, destination] {
                if resolved.attachment.is_some() {
                    needed
                        .entry(resolved.allocation.clone())
                        .and_modify(|known| known.writable |= resolved.writable)
                        .or_insert(resolved);
                }
            }
        }

        for (allocation, resolved) in needed {
            self.give(transfer_server, allocation, &resolved);
        }
        let task = self.transfers.next_task;
        self.transfers.next_task += 1;
        self.transfers.tasks.insert(
            task,
            Task {
                client,
                deployment: deployment.to_owned(),
                update,
                priority,
                bytes,
                copied: 0,
                cancelled: false,
            },
        );
        self.events.push((
            transfer_server,
            Event::Transfer {
                task,
                priority,
                moves,
            },
            Vec::new(),
        ));
        Ok(task)
    }

    /// Give the Transfer Server `allocation` with its descriptor, unless it
    /// has it already.
    fn give(&mut self, transfer_server: ClientId, allocation: Allocation, resolved: &Resolved) {
        if !self.transfers.given.insert(allocation.clone()) {
            return;
        }
        let Some(attachment) = resolved.attachment else {
            return;
        };
        self.events.push((
            transfer_server,
            Event::Allocation {
                allocation,
                memory: resolved.memory,
                bytes: resolved.bytes,
                writable: resolved.writable,
            },
            vec![attachment],
        ));
    }

    /// The allocation `region` names, for the caller's `deployment`; a
    /// `destination` is written, so it is no read-only tensor.
    fn resolve(
        &self,
        deployment: &str,
        region: &Region,
        destination: bool,
    ) -> Result<Resolved, (ErrorKind, String)> {
        match region {
            Region::Tensor { key } => {
                let tensor = self
                    .tensors
                    .get(key)
                    .filter(|tensor| {
                        tensor
                            .retainers
                            .iter()
                            .any(|retainer| retainer.scope.deployment == deployment)
                    })
                    .ok_or_else(|| {
                        invalid(&format!("{key:?} is no retained tensor of {deployment}"))
                    })?;
                if destination && tensor.access == Access::ReadOnly {
                    return Err(invalid(&format!(
                        "{} is read-only and is not written",
                        tensor.name
                    )));
                }
                let storage = self
                    .storages
                    .get(&tensor.storage)
                    .ok_or_else(|| invalid(&format!("{} has no storage", tensor.name)))?;
                let memory = match storage.record.placement {
                    Placement::Device { gpu } => Memory::Device { gpu },
                    Placement::Host { .. } => Memory::Host,
                };
                Ok(Resolved {
                    allocation: Allocation::Storage {
                        storage: tensor.storage,
                    },
                    memory,
                    base: tensor.offset,
                    bytes: storage.record.bytes,
                    attachment: Some(Attachment::Storage(tensor.storage)),
                    writable: self.writable(tensor.storage),
                })
            }
            Region::Pool { pool } => {
                let entry = self
                    .pages
                    .get(*pool)
                    .filter(|entry| entry.deployment == deployment && !self.pages.released(*pool))
                    .ok_or_else(|| {
                        invalid(&format!(
                            "page pool {pool} is not a live pool of {deployment}"
                        ))
                    })?;
                Ok(Resolved {
                    allocation: Allocation::Pool { pool: *pool },
                    memory: Memory::Host,
                    base: 0,
                    bytes: entry.slots() * entry.shape.page_bytes,
                    attachment: Some(Attachment::Pool(*pool)),
                    writable: true,
                })
            }
            Region::Fabric {
                handle,
                gpu,
                bytes,
                offset,
            } => {
                if handle.len() != 64 {
                    return Err(invalid("a fabric handle has 64 bytes"));
                }
                Ok(Resolved {
                    allocation: Allocation::Fabric {
                        handle: handle.clone(),
                        gpu: *gpu,
                        bytes: *bytes,
                    },
                    memory: Memory::Device { gpu: *gpu },
                    base: *offset,
                    bytes: *bytes,
                    attachment: None,
                    writable: destination,
                })
            }
        }
    }

    /// Whether the Transfer Server maps `storage` for writing: it holds an
    /// exclusive tensor. Read-only tensors are immutable, so a storage of
    /// read-only tensors alone is mapped read-only.
    fn writable(&self, storage: StorageId) -> bool {
        self.tensors
            .values()
            .any(|tensor| tensor.storage == storage && tensor.access == Access::Exclusive)
    }

    /// Cancel every task `selected` picks that is not cancelled yet;
    /// returns how many.
    fn cancel_where(&mut self, selected: impl Fn(u64, &Task) -> bool) -> u64 {
        let mut cancelled = 0;
        for (&id, task) in &mut self.transfers.tasks {
            if task.cancelled || !selected(id, task) {
                continue;
            }
            task.cancelled = true;
            cancelled += 1;
            if let Some(transfer_server) = self.transfers.client {
                self.events
                    .push((transfer_server, Event::Cancel { task: id }, Vec::new()));
            }
        }
        cancelled
    }

    /// See [`Request::Throttle`](storage::protocol::Request::Throttle).
    fn throttle(&mut self, links: &[Link], throttle: Throttle) -> Result<(), (ErrorKind, String)> {
        let Some(transfer_server) = self.transfers.client else {
            return Err(unavailable());
        };
        self.events.push((
            transfer_server,
            Event::Throttle {
                links: links.to_vec(),
                throttle,
            },
            Vec::new(),
        ));
        Ok(())
    }

    /// The Transfer Server reported task `task`.
    fn copied(&mut self, task: u64, bytes: u64, end: Option<TransferEnd>) {
        let Some(entry) = self.transfers.tasks.get_mut(&task) else {
            return;
        };
        entry.copied = bytes;
        let client = entry.client;
        let event = match end {
            Some(end) => {
                self.transfers.tasks.remove(&task);
                Event::Transferred { task, bytes, end }
            }
            None => Event::TransferProgress { task, bytes },
        };
        self.events.push((client, event, Vec::new()));
    }

    /// A client's connection closed: the tasks it submitted are cancelled,
    /// since nobody waits for them any more.
    pub(super) fn transfer_client_gone(&mut self, id: ClientId) {
        let cancelled = self.cancel_where(|_, task| task.client == id);
        if cancelled > 0 {
            tracing::info!(
                tasks = cancelled,
                "cancelled the transfers of a client that disconnected"
            );
        }
    }

    /// The Transfer Server's connection closed: every task it held fails,
    /// and the next one is given the allocations of new tasks afresh.
    pub(super) fn transfer_server_gone(&mut self) {
        tracing::warn!("the Transfer Server disconnected");
        self.transfers.client = None;
        self.transfers.given.clear();
        self.transfers.links.clear();
        self.transfers.contexts.clear();
        for (task, entry) in std::mem::take(&mut self.transfers.tasks) {
            self.events.push((
                entry.client,
                Event::Transferred {
                    task,
                    bytes: entry.copied,
                    end: TransferEnd::Failed {
                        message: "the Transfer Server ended".into(),
                    },
                },
                Vec::new(),
            ));
        }
    }

    /// The server released `allocation`: the Transfer Server, if it was
    /// given it, unmaps it once no task copies it.
    pub(super) fn allocation_released(&mut self, allocation: Allocation) {
        if !self.transfers.given.remove(&allocation) {
            return;
        }
        if let Some(transfer_server) = self.transfers.client {
            self.events.push((
                transfer_server,
                Event::AllocationReleased { allocation },
                Vec::new(),
            ));
        }
    }

    /// The Transfer Server in a listing, while it is connected.
    pub(super) fn listed_transfer(&self) -> Option<ListedTransfer> {
        let client = self.clients.get(&self.transfers.client?)?;
        Some(ListedTransfer {
            pid: client.pid,
            links: self.transfers.links.clone(),
            contexts: self.transfers.contexts.clone(),
            tasks: self
                .transfers
                .tasks
                .iter()
                .map(|(&task, entry)| ListedTask {
                    task,
                    deployment: entry.deployment.clone(),
                    update: entry.update,
                    priority: entry.priority,
                    bytes: entry.bytes,
                    copied: entry.copied,
                    cancelled: entry.cancelled,
                })
                .collect(),
        })
    }
}

/// The offset in `resolved`'s allocation of a span's `offset`, once the
/// span's `bytes` from there lie within the allocation.
fn within(
    resolved: &Resolved,
    offset: u64,
    bytes: u64,
    side: &str,
) -> Result<u64, (ErrorKind, String)> {
    let start = resolved.base.checked_add(offset);
    match start.and_then(|start| start.checked_add(bytes).map(|end| (start, end))) {
        Some((start, end)) if end <= resolved.bytes => Ok(start),
        _ => Err(invalid(&format!(
            "a span's {side} of {bytes} bytes at offset {offset} reaches past its region's {} \
             bytes from offset {}",
            resolved.bytes, resolved.base
        ))),
    }
}

fn unavailable() -> (ErrorKind, String) {
    (
        ErrorKind::Unavailable,
        "no Transfer Server is connected".into(),
    )
}

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
//! The server submits tasks of its own too: a world's weights moving
//! between its GPUs and host memory ([`crate::moves`]) copy whole storages,
//! and each task's end goes to its move rather than to a client.
//!
//! A span's source on another host ([`Region::Remote`]) is forwarded as
//! that host's region ([`Allocation::Remote`]), which this server cannot
//! check: the Transfer Server has that host's server expose it and READs
//! it. The other direction serves another host's Transfer Server, a
//! [`Role::Peer`]: its [`Request::Expose`](storage::protocol::Request::Expose)
//! names regions of a deployment here, which the server checks as a
//! task's sources, gives the Transfer Server, and has it register with its
//! NIXL agent ([`Event::Register`]); the peer's request waits until the
//! Transfer Server reports the ranges and its agent's metadata
//! ([`Request::Registered`](storage::protocol::Request::Registered)), which
//! answer it ([`Reply::Exposed`]). The exposure lives until the peer ends it
//! ([`Request::Unexpose`](storage::protocol::Request::Unexpose)) or its
//! connection closes; the Transfer Server then lets its registrations go
//! ([`Event::Deregister`]), and the GPU contexts they held with them.
//!
//! When the Transfer Server's connection closes, every task it held fails
//! and the submitters learn it at once: the update a task served aborts,
//! and the next Transfer Server maps the allocations of new tasks again. An
//! exposure waiting for it is refused, and the next Transfer Server's agent
//! registers the regions afresh when a peer asks again. While none is
//! connected a submission or an exposure is refused
//! ([`ErrorKind::Unavailable`]).

use std::collections::{BTreeMap, BTreeSet};

use storage::protocol::{
    Access, Allocation, ContextBytes, ErrorKind, Event, FabricExport, FabricHandle, Link,
    LinkBandwidth, ListedTask, ListedTransfer, Memory, Move, Placement, Priority, Region,
    Registered, Reply, Role, Span, Throttle, TransferEnd,
};

use crate::child::{Launch, Supervised};
use crate::{Attachment, ClientId, Outcome, Server, StorageId, invalid};

/// A task the Transfer Server runs, as the server keeps it.
pub(super) struct Task {
    /// Who learns of the task's progress and end.
    owner: Owner,
    deployment: String,
    update: u64,
    priority: Priority,
    /// The bytes of all its spans, and those copied so far.
    bytes: u64,
    copied: u64,
    cancelled: bool,
}

/// Who submitted a task, and learns of its progress and end.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Owner {
    /// A head or a worker, which receives its events.
    Client(ClientId),
    /// Item `index` of the server's own weight move `number`.
    Move { number: u64, index: usize },
}

/// The Transfer Server process, its connection and its tasks.
pub(super) struct Transfers {
    pub(super) process: Supervised,
    /// The connected Transfer Server's client.
    pub(super) client: Option<ClientId>,
    pub(super) tasks: BTreeMap<u64, Task>,
    next_task: u64,
    sessions: BTreeMap<u64, ClientId>,
    next_session: u64,
    /// The allocations the connected Transfer Server was given.
    given: BTreeSet<Allocation>,
    /// What it measured: every link of this host when it started, and the
    /// links from other hosts it timed since.
    links: Vec<LinkBandwidth>,
    contexts: Vec<ContextBytes>,
    /// Peers' exposures waiting for the Transfer Server's registrations, by
    /// the number of their [`Event::Register`].
    exposures: BTreeMap<u64, Exposure>,
    next_registration: u64,
    /// Exposures registered and not ended, by that number, with the peer
    /// that holds each.
    live: BTreeMap<u64, ClientId>,
}

impl Transfers {
    pub(super) fn new(launch: Option<Launch>) -> Self {
        Self {
            process: Supervised::new("Transfer Server", launch),
            client: None,
            tasks: BTreeMap::new(),
            next_task: 1,
            sessions: BTreeMap::new(),
            next_session: 1,
            given: BTreeSet::new(),
            links: Vec::new(),
            contexts: Vec::new(),
            exposures: BTreeMap::new(),
            next_registration: 1,
            live: BTreeMap::new(),
        }
    }
}

/// The regions of a peer's exposure, checked and resolved: the allocations
/// the Transfer Server registers, each once, and for each region the index
/// of its allocation among them and the offset of its first byte there.
pub(super) struct Exposing {
    allocations: Vec<Allocation>,
    regions: Vec<(usize, u64)>,
    /// Each allocation's fabric handle and length, when its creator
    /// recorded one.
    fabric: Vec<Option<(FabricHandle, u64)>>,
}

/// A peer's exposure the server answers once its Transfer Server reported
/// the registrations: the peer's client and request, and its regions.
struct Exposure {
    client: ClientId,
    request: u64,
    regions: Vec<(usize, u64)>,
    fabric: Vec<Option<(FabricHandle, u64)>>,
}

impl Exposure {
    /// The fabric handle of a region at `base` in allocation `index`, as
    /// the reader imports it: the whole allocation, and the region's
    /// offset in it.
    fn fabric_at(&self, index: usize, base: u64) -> Option<FabricExport> {
        let (handle, bytes) = self.fabric.get(index).copied().flatten()?;
        Some(FabricExport {
            handle,
            bytes,
            offset: base,
        })
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
            Request::OpenTransfer { gpus } => {
                let transfer = self
                    .transfers
                    .client
                    .ok_or_else(|| (ErrorKind::Unavailable, "no Transfer Server runs".into()))?;
                if gpus.iter().any(|gpu| {
                    !self
                        .transfers
                        .contexts
                        .iter()
                        .any(|context| context.gpu == *gpu)
                }) {
                    return Err(invalid("a transfer session requires local GPUs"));
                }
                let session = self.transfers.next_session;
                self.transfers.next_session += 1;
                self.transfers.sessions.insert(session, id);
                self.events.push((
                    transfer,
                    Event::TransferSession {
                        session,
                        gpus: gpus.clone(),
                    },
                    Vec::new(),
                ));
                Ok(Outcome::Reply(
                    Reply::TransferOpened { session },
                    Vec::new(),
                ))
            }
            Request::CloseTransfer { session } => {
                if self.transfers.sessions.get(session) != Some(&id) {
                    return Err(invalid("the connection owns no such transfer session"));
                }
                self.close_transfer(*session);
                Ok(Outcome::Reply(Reply::TransferClosed, Vec::new()))
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
                self.throttle(&deployment, links, *throttle)?;
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
                // A later report adds links from other hosts as copies
                // timed them; the latest measurement of a link stands.
                let known = &mut self.transfers.links;
                for measured in links {
                    match known.iter_mut().find(|entry| entry.link == measured.link) {
                        Some(entry) => entry.clone_from(measured),
                        None => known.push(measured.clone()),
                    }
                }
                let known = &mut self.transfers.contexts;
                for context in contexts {
                    tracing::info!(
                        gpu = %cuda::DeviceUuid(context.gpu),
                        bytes = context.bytes,
                        "the Transfer Server's CUDA context"
                    );
                    match known.iter_mut().find(|entry| entry.gpu == context.gpu) {
                        Some(entry) => *entry = *context,
                        None => known.push(*context),
                    }
                }
                Ok(Outcome::Reply(Reply::Noted, Vec::new()))
            }
            Request::Copied { task, bytes, end } => {
                self.copied(*task, *bytes, end.clone());
                Ok(Outcome::Reply(Reply::Noted, Vec::new()))
            }
            Request::Registered {
                request,
                metadata,
                ranges,
                failure,
            } => {
                self.registered(*request, metadata, ranges, failure.as_deref());
                Ok(Outcome::Reply(Reply::Noted, Vec::new()))
            }
            _ => Err(invalid("not a request of the Transfer Server")),
        }
    }

    /// A peer's [`Request::Expose`](storage::protocol::Request::Expose) of
    /// `regions` of `deployment`: each is a retained tensor or a page pool
    /// of the deployment, checked as a task's source is, and the Transfer
    /// Server is given each allocation it does not hold yet. The request
    /// then waits for the registrations ([`Server::begin_exposure`]).
    pub(super) fn expose(
        &mut self,
        deployment: &str,
        regions: &[Region],
    ) -> Result<Outcome, (ErrorKind, String)> {
        let Some(transfer_server) = self.transfers.client else {
            return Err(unavailable());
        };
        if regions.is_empty() {
            return Err(invalid("an exposure names at least one region"));
        }

        let mut allocations: Vec<Allocation> = Vec::new();
        let mut placed = Vec::with_capacity(regions.len());
        let mut needed = BTreeMap::new();
        for region in regions {
            if !matches!(region, Region::Tensor { .. } | Region::Pool { .. }) {
                return Err(invalid(
                    "a host exposes its own retained tensors and page pools",
                ));
            }
            let resolved = self.resolve(deployment, region, false)?;
            let index = match allocations
                .iter()
                .position(|known| *known == resolved.allocation)
            {
                Some(index) => index,
                None => {
                    allocations.push(resolved.allocation.clone());
                    allocations.len() - 1
                }
            };
            placed.push((index, resolved.base));
            need(&mut needed, resolved);
        }
        for (allocation, resolved) in needed {
            self.give(transfer_server, allocation, &resolved);
        }
        // A storage's creator may have recorded its fabric handle, which a
        // reader of the same fabric prefers to a READ.
        let fabric = allocations
            .iter()
            .map(|allocation| match allocation {
                Allocation::Storage { storage } => self.storages.get(storage).and_then(|entry| {
                    entry
                        .record
                        .fabric
                        .map(|handle| (handle, entry.record.bytes))
                }),
                _ => None,
            })
            .collect();
        Ok(Outcome::Exposing(Exposing {
            allocations,
            regions: placed,
            fabric,
        }))
    }

    /// Have the Transfer Server register an exposure's allocations, and hold
    /// the connection of `client`, the peer, until it reports them and
    /// `request` is answered: requests are answered in order.
    pub(super) fn begin_exposure(&mut self, client: ClientId, request: u64, exposing: Exposing) {
        let Some(transfer_server) = self.transfers.client else {
            self.answer(client, request, error_reply(unavailable()), &[]);
            return;
        };
        if let Some(entry) = self.clients.get_mut(&client) {
            entry.held = true;
        }
        let number = self.transfers.next_registration;
        self.transfers.next_registration += 1;
        tracing::debug!(
            exposure = number,
            allocations = exposing.allocations.len(),
            "registering allocations for another host"
        );
        self.events.push((
            transfer_server,
            Event::Register {
                request: number,
                allocations: exposing.allocations,
            },
            Vec::new(),
        ));
        self.transfers.exposures.insert(
            number,
            Exposure {
                client,
                request,
                regions: exposing.regions,
                fabric: exposing.fabric,
            },
        );
    }

    /// The Transfer Server registered the allocations of exposure
    /// `number`, or failed to: the peer's request is answered, and its
    /// connection's next requests are served.
    fn registered(
        &mut self,
        number: u64,
        metadata: &[u8],
        ranges: &[Registered],
        failure: Option<&str>,
    ) {
        tracing::debug!(
            exposure = number,
            failure,
            "the Transfer Server registered an exposure"
        );
        let Some(exposure) = self.transfers.exposures.remove(&number) else {
            // The peer disconnected meanwhile: nobody reads what was
            // registered for it.
            if failure.is_none() {
                self.deregister(number);
            }
            return;
        };
        let reply = match failure {
            Some(message) => error_reply((ErrorKind::Resource, message.to_owned())),
            None => {
                // Each region's range runs from its first byte to the end
                // of its allocation.
                let regions = exposure
                    .regions
                    .iter()
                    .map(|&(index, base)| {
                        let range = ranges.get(index).filter(|range| base <= range.bytes)?;
                        Some(Registered {
                            memory: range.memory.clone(),
                            key: range.key,
                            address: range.address + base,
                            bytes: range.bytes - base,
                            fabric: exposure.fabric_at(index, base),
                        })
                    })
                    .collect::<Option<Vec<_>>>();
                match regions {
                    Some(regions) => {
                        self.transfers.live.insert(number, exposure.client);
                        Reply::Exposed {
                            exposure: number,
                            metadata: metadata.to_vec(),
                            regions,
                        }
                    }
                    None => {
                        self.deregister(number);
                        error_reply((
                            ErrorKind::Resource,
                            "the Transfer Server reported no range for an exposed region".into(),
                        ))
                    }
                }
            }
        };
        self.release_exposure(exposure, reply);
    }

    /// A peer's [`Request::Unexpose`](storage::protocol::Request::Unexpose):
    /// its exposure `exposure` ends, unless it holds no such exposure.
    pub(super) fn unexpose(&mut self, peer: ClientId, exposure: u64) -> Outcome {
        if self.transfers.live.get(&exposure) == Some(&peer) {
            self.transfers.live.remove(&exposure);
            self.deregister(exposure);
        }
        Outcome::Reply(Reply::Unexposed, Vec::new())
    }

    /// Have the Transfer Server let the registrations of exposure `number`
    /// go.
    fn deregister(&mut self, number: u64) {
        if let Some(transfer_server) = self.transfers.client {
            self.events.push((
                transfer_server,
                Event::Deregister { request: number },
                Vec::new(),
            ));
        }
    }

    /// Answer a peer's exposure with `reply` and serve its connection on.
    fn release_exposure(&mut self, exposure: Exposure, reply: Reply) {
        self.answer(exposure.client, exposure.request, reply, &[]);
        if let Some(client) = self.clients.get_mut(&exposure.client) {
            client.held = false;
        }
        self.serve_queue(exposure.client);
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
            if span.bytes == 0 || span.rows == 0 {
                return Err(invalid("a span copies at least one byte"));
            }
            // Rows at a pitch: each pitch holds a row, and every row lies
            // in the span's regions, which its extent from the first row's
            // start to the last row's end checks.
            if span.rows > 1
                && (span.source_pitch < span.bytes || span.destination_pitch < span.bytes)
            {
                return Err(invalid(
                    "a span's rows overlap: a pitch is shorter than a row",
                ));
            }
            let extent = |pitch: u64| {
                (span.rows - 1)
                    .checked_mul(pitch)
                    .and_then(|start| start.checked_add(span.bytes))
                    .ok_or_else(|| invalid("a span's rows overflow"))
            };
            let destination = self.resolve(deployment, &span.destination, true)?;
            let destination_offset = within(
                &destination,
                span.destination_offset,
                extent(span.destination_pitch)?,
                "destination",
            )?;
            let crossing =
                invalid("a span copies to or from this host's memory, not between two other hosts");
            let (source, source_offset) = match &span.source {
                // Another host's region: that host checks it, and the span
                // against it, once the Transfer Server asks it to expose
                // the region.
                Region::Remote { endpoint, region } => {
                    if matches!(destination.allocation, Allocation::Fabric { .. }) {
                        return Err(crossing);
                    }
                    (remote(deployment, endpoint, region)?, span.source_offset)
                }
                region => {
                    let source = self.resolve(deployment, region, false)?;
                    if matches!(source.allocation, Allocation::Fabric { .. })
                        && matches!(destination.allocation, Allocation::Fabric { .. })
                    {
                        return Err(crossing);
                    }
                    let offset = within(
                        &source,
                        span.source_offset,
                        extent(span.source_pitch)?,
                        "source",
                    )?;
                    let allocation = source.allocation.clone();
                    need(&mut needed, source);
                    (allocation, offset)
                }
            };
            bytes = span
                .bytes
                .checked_mul(span.rows)
                .and_then(|copied| bytes.checked_add(copied))
                .ok_or_else(|| invalid("the spans' bytes overflow"))?;
            moves.push(Move {
                source,
                source_offset,
                destination: destination.allocation.clone(),
                destination_offset,
                bytes: span.bytes,
                rows: span.rows,
                source_pitch: span.source_pitch,
                destination_pitch: span.destination_pitch,
            });
            need(&mut needed, destination);
        }

        for (allocation, resolved) in needed {
            self.give(transfer_server, allocation, &resolved);
        }
        let task = self.transfers.next_task;
        self.transfers.next_task += 1;
        self.transfers.tasks.insert(
            task,
            Task {
                owner: Owner::Client(client),
                deployment: deployment.to_owned(),
                update,
                priority,
                bytes,
                copied: 0,
                cancelled: false,
            },
        );
        tracing::info!(
            task,
            spans = moves.len(),
            bytes,
            ?priority,
            "forwarding a transfer task"
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

    /// Have the Transfer Server copy the whole of storage `source` into
    /// storage `target`, `bytes` of each, for item `index` of the weight
    /// move `number` of `deployment` ([`crate::moves`]), which learns of its
    /// end ([`Server::moved`]). Weight moves wait for no promotion.
    pub(super) fn copy_storage(
        &mut self,
        deployment: &str,
        (source, target): (StorageId, StorageId),
        bytes: u64,
        (number, index): (u64, usize),
    ) -> Result<(), (ErrorKind, String)> {
        let Some(transfer_server) = self.transfers.client else {
            return Err(unavailable());
        };
        let (Some(from), Some(to)) = (self.storages.get(&source), self.storages.get(&target))
        else {
            return Err(invalid("a moved storage was released"));
        };
        let resolved = |storage: StorageId, record: &storage::protocol::StorageRecord| Resolved {
            allocation: Allocation::Storage { storage },
            memory: crate::moves::memory_of(record),
            base: 0,
            bytes: record.bytes,
            attachment: Some(Attachment::Storage(storage)),
            writable: false,
        };
        let from = resolved(source, &from.record);
        let to = Resolved {
            writable: true,
            ..resolved(target, &to.record)
        };
        // One row of the whole storage; its pitches are unused.
        let moves = vec![Move {
            source: from.allocation.clone(),
            source_offset: 0,
            destination: to.allocation.clone(),
            destination_offset: 0,
            bytes,
            rows: 1,
            source_pitch: bytes,
            destination_pitch: bytes,
        }];
        self.give(transfer_server, from.allocation.clone(), &from);
        self.give(transfer_server, to.allocation.clone(), &to);

        let task = self.transfers.next_task;
        self.transfers.next_task += 1;
        self.transfers.tasks.insert(
            task,
            Task {
                owner: Owner::Move { number, index },
                deployment: deployment.to_owned(),
                update: 0,
                priority: Priority::Background,
                bytes,
                copied: 0,
                cancelled: false,
            },
        );
        self.events.push((
            transfer_server,
            Event::Transfer {
                task,
                priority: Priority::Background,
                moves,
            },
            Vec::new(),
        ));
        Ok(())
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
                memory: resolved.memory.clone(),
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
            Region::Tensor { key, at } => {
                // The copy at `at`, or the key's only copy.
                let copies: Vec<_> = self
                    .copies_of(key, at.as_slice())
                    .into_iter()
                    .filter_map(|id| self.tensors.get(&id))
                    .filter(|tensor| {
                        tensor
                            .retainers
                            .iter()
                            .any(|retainer| retainer.scope.deployment == deployment)
                    })
                    .collect();
                let tensor = match copies.as_slice() {
                    [tensor] => *tensor,
                    [] => {
                        return Err(invalid(&format!(
                            "{key:?} is no retained tensor of {deployment}"
                        )));
                    }
                    _ => {
                        return Err(invalid(&format!(
                            "{key:?} has a copy at several locations; a region names one"
                        )));
                    }
                };
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
            Region::Remote { .. } => Err(invalid(if destination {
                "a span writes memory of this host: another host's region is a source, which \
                 this host READs"
            } else {
                "another host exposes its own retained tensors and page pools"
            })),
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
    /// Without links, a halving names the tasks of `deployment`, whose links
    /// the Transfer Server knows.
    fn throttle(
        &mut self,
        deployment: &str,
        links: &[Link],
        throttle: Throttle,
    ) -> Result<(), (ErrorKind, String)> {
        let Some(transfer_server) = self.transfers.client else {
            return Err(unavailable());
        };
        let tasks = if links.is_empty() && throttle == Throttle::Halve {
            self.transfers
                .tasks
                .iter()
                .filter(|(_, task)| task.deployment == deployment)
                .map(|(&id, _)| id)
                .collect()
        } else {
            Vec::new()
        };
        self.events.push((
            transfer_server,
            Event::Throttle {
                links: links.to_vec(),
                throttle,
                tasks,
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
        let owner = entry.owner;
        if end.is_some() {
            self.transfers.tasks.remove(&task);
        }
        match (owner, end) {
            (Owner::Client(client), Some(end)) => {
                self.events
                    .push((client, Event::Transferred { task, bytes, end }, Vec::new()));
            }
            (Owner::Client(client), None) => {
                self.events
                    .push((client, Event::TransferProgress { task, bytes }, Vec::new()));
            }
            (Owner::Move { number, index }, Some(end)) => self.moved(number, index, &end),
            (Owner::Move { .. }, None) => {}
        }
    }

    fn close_transfer(&mut self, session: u64) {
        self.transfers.sessions.remove(&session);
        if let Some(transfer) = self.transfers.client {
            self.events.push((
                transfer,
                Event::TransferSession {
                    session,
                    gpus: Vec::new(),
                },
                Vec::new(),
            ));
        }
    }

    /// A client's connection closed: the tasks it submitted are cancelled,
    /// since nobody waits for them any more.
    pub(super) fn transfer_client_gone(&mut self, id: ClientId) {
        let sessions: Vec<_> = self
            .transfers
            .sessions
            .iter()
            .filter(|(_, owner)| **owner == id)
            .map(|(&session, _)| session)
            .collect();
        for session in sessions {
            self.close_transfer(session);
        }
        // A peer that left waits for no exposure, and reads none any more:
        // the registrations its exposures held go.
        self.transfers
            .exposures
            .retain(|_, exposure| exposure.client != id);
        let ended: Vec<u64> = self
            .transfers
            .live
            .iter()
            .filter(|(_, peer)| **peer == id)
            .map(|(number, _)| *number)
            .collect();
        for number in ended {
            self.transfers.live.remove(&number);
            self.deregister(number);
        }
        let cancelled = self.cancel_where(|_, task| task.owner == Owner::Client(id));
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
        self.transfers.sessions.clear();
        let failed = TransferEnd::Failed {
            message: "the Transfer Server ended".into(),
        };
        for (task, entry) in std::mem::take(&mut self.transfers.tasks) {
            match entry.owner {
                Owner::Client(client) => self.events.push((
                    client,
                    Event::Transferred {
                        task,
                        bytes: entry.copied,
                        end: failed.clone(),
                    },
                    Vec::new(),
                )),
                Owner::Move { number, index } => self.moved(number, index, &failed),
            }
        }
        // The device memory it was creating for a move is never reported.
        self.moves_lost_allocations();
        // Its agent ended with it: the exposures it held end, the regions it
        // was registering are refused, and the next one registers them when
        // a peer asks again.
        self.transfers.live.clear();
        for (_, exposure) in std::mem::take(&mut self.transfers.exposures) {
            let reply = error_reply((
                ErrorKind::Unavailable,
                "the Transfer Server ended before it registered the regions".into(),
            ));
            self.release_exposure(exposure, reply);
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

    /// The device memory the connected Transfer Server's CUDA context takes
    /// on each GPU while it holds one, as it measured it; it books the
    /// context itself while it holds it.
    pub(super) fn transfer_contexts(&self) -> Vec<([u8; 16], u64)> {
        self.transfers
            .contexts
            .iter()
            .map(|context| (context.gpu, context.bytes))
            .collect()
    }

    /// The bandwidth of copies from host memory to `gpu` the connected
    /// Transfer Server measured, in bytes per second.
    pub(super) fn host_link(&self, gpu: [u8; 16]) -> Option<u64> {
        let link = Link {
            source: Memory::Host,
            destination: Memory::Device { gpu },
        };
        self.transfers
            .links
            .iter()
            .find(|measured| measured.link == link)
            .map(|measured| measured.bytes_per_second)
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

/// Note that the Transfer Server needs `resolved`'s descriptor, mapped for
/// writing when any span writes it.
fn need(needed: &mut BTreeMap<Allocation, Resolved>, resolved: Resolved) {
    if resolved.attachment.is_some() {
        needed
            .entry(resolved.allocation.clone())
            .and_modify(|known| known.writable |= resolved.writable)
            .or_insert(resolved);
    }
}

/// The allocation of `region`, of `deployment`, at the Storage Server of
/// another host listening at `endpoint`: a retained tensor or a page pool
/// there, which that host checks once the Transfer Server asks it to expose
/// the region.
fn remote(
    deployment: &str,
    endpoint: &str,
    region: &Region,
) -> Result<Allocation, (ErrorKind, String)> {
    if !matches!(region, Region::Tensor { .. } | Region::Pool { .. }) {
        return Err(invalid(
            "another host's region is one of its retained tensors or page pools",
        ));
    }
    if endpoint.is_empty() {
        return Err(invalid(
            "another host's region names the endpoint of its Storage Server",
        ));
    }
    Ok(Allocation::Remote {
        endpoint: endpoint.to_owned(),
        deployment: deployment.to_owned(),
        region: Box::new(region.clone()),
    })
}

fn error_reply((kind, message): (ErrorKind, String)) -> Reply {
    Reply::Error { kind, message }
}

pub(super) fn unavailable() -> (ErrorKind, String) {
    (
        ErrorKind::Unavailable,
        "no Transfer Server is connected".into(),
    )
}

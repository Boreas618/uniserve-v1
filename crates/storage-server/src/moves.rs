//! Moving a world's weights between its GPUs and host storage: a previous
//! world kept in host storage on trial (`Request::Offload`), and returned
//! to its GPUs before it starts again at a rollback (`Request::Restore`).
//!
//! The server never copies a byte itself: its Transfer Server does. An
//! offload takes every device storage of read-only tensors that the world
//! alone retains, creates host storage of each one's size (admitted under
//! the host rule as one, so a refused offload moves nothing), and has the
//! Transfer Server copy each storage there. A restore takes every host
//! storage an offload of the world left, has the Transfer Server create
//! device storage of each one's size on the GPU it came from (admitted
//! against that GPU's ledger first) and copy it back. Only once every copy
//! of a move ended do its tensors lie in the copies and the storages they
//! left go: each GPU's ledger then books the move's effect at once, the
//! device storage an offload freed for the other world, the device storage a
//! restore took for the world that starts again. A copy that fails leaves
//! every tensor where it was and releases the copies made so far.
//!
//! The head's request is answered when its move ends, and holds its
//! connection meanwhile, as an exposure does.

use std::collections::BTreeMap;
use std::os::fd::OwnedFd;

use ::storage::protocol::{
    Access, Allocation, ErrorKind, Event, LinkEnd, Placement, Reply, StorageRecord, TransferEnd,
};

use crate::{ClientId, Outcome, Server, StorageEntry, StorageId, invalid};

/// The moves under way and the Transfer Server's allocations they wait
/// for.
#[derive(Default)]
pub(crate) struct Moves {
    next: u64,
    active: BTreeMap<u64, Moving>,
    /// The Transfer Server's allocations of a restore under way: the
    /// allocation request's number, with the move and the item.
    allocating: BTreeMap<u64, (u64, usize)>,
    next_allocation: u64,
}

/// A move a head asked for, checked: its deployment, its direction and
/// its storages.
pub(crate) struct Planned {
    deployment: String,
    to_host: bool,
    items: Vec<Item>,
}

/// A move under way: the head waiting for it, and its storages.
struct Moving {
    client: ClientId,
    request: u64,
    deployment: String,
    to_host: bool,
    items: Vec<Item>,
    /// Why the move failed, from the first copy or allocation that did.
    failure: Option<String>,
}

/// One storage a move copies.
struct Item {
    source: StorageId,
    /// The GPU the storage lies on (an offload) or returns to (a restore).
    gpu: [u8; 16],
    bytes: u64,
    /// The copy, once its storage exists.
    target: Option<StorageId>,
    /// Whether its copy ended, well or not.
    ended: bool,
}

impl Server {
    /// Checks a head's [`Request::Offload`] (`to_host`) or
    /// [`Request::Restore`] of `world` of `deployment` and picks its
    /// storages: answered at once when there is none, refused when the
    /// host (an offload) or a GPU's ledger (a restore) cannot hold them,
    /// or when no Transfer Server runs.
    ///
    /// [`Request::Offload`]: ::storage::protocol::Request::Offload
    /// [`Request::Restore`]: ::storage::protocol::Request::Restore
    pub(crate) fn plan_move(
        &self,
        id: ClientId,
        deployment: &str,
        world: u64,
        to_host: bool,
    ) -> Result<Outcome, (ErrorKind, String)> {
        if !self
            .deployments
            .get(deployment)
            .is_some_and(|entry| entry.head == Some(id))
        {
            return Err(invalid(&format!(
                "only the head that claimed {deployment} moves its worlds' weights"
            )));
        }
        // The storages of read-only tensors the world alone retains (an
        // offload), or that an offload of the world left in host storage (a
        // restore).
        let mut candidates: BTreeMap<StorageId, bool> = BTreeMap::new();
        for tensor in self.tensors.values() {
            let alone = tensor.access == Access::ReadOnly
                && !tensor.retainers.is_empty()
                && tensor.retainers.iter().all(|retainer| {
                    retainer.scope.deployment == deployment && retainer.world == world
                });
            let entry = candidates.entry(tensor.storage).or_insert(true);
            *entry &= alone;
        }
        let mut items = Vec::new();
        for (storage, alone) in candidates {
            let Some(entry) = self.storages.get(&storage) else {
                continue;
            };
            let gpu = match (to_host, entry.record.placement, entry.moved_from) {
                (true, Placement::Device { gpu }, _) if alone => gpu,
                (false, Placement::Host { .. }, Some(gpu)) if alone => gpu,
                _ => continue,
            };
            items.push(Item {
                source: storage,
                gpu,
                bytes: entry.record.bytes,
                target: None,
                ended: false,
            });
        }
        if items.is_empty() {
            return Ok(Outcome::Reply(
                Reply::Moved {
                    storages: 0,
                    bytes: 0,
                },
                Vec::new(),
            ));
        }
        if self.transfers.client.is_none() {
            return Err(crate::transfers::unavailable());
        }
        if to_host {
            self.admit_host(items.iter().map(|item| item.bytes).sum())?;
        } else {
            let mut by_gpu: BTreeMap<[u8; 16], u64> = BTreeMap::new();
            for item in &items {
                *by_gpu.entry(item.gpu).or_default() += item.bytes;
            }
            for (gpu, bytes) in by_gpu {
                self.admit_device(deployment, gpu, bytes, &format!("world {world}'s weights"))?;
            }
        }
        Ok(Outcome::Moving(Planned {
            deployment: deployment.to_owned(),
            to_host,
            items,
        }))
    }

    /// Starts the checked move `planned` for the head `client`, whose
    /// `request` is answered when it ends: an offload creates each host
    /// copy and has the Transfer Server copy into it; a restore has the
    /// Transfer Server create each device copy first ([`Self::allocated`]).
    pub(crate) fn begin_move(&mut self, client: ClientId, request: u64, planned: Planned) {
        if let Some(entry) = self.clients.get_mut(&client) {
            entry.held = true;
        }
        let number = self.moves.next;
        self.moves.next += 1;
        let Planned {
            deployment,
            to_host,
            items,
        } = planned;
        let count = items.len();
        self.moves.active.insert(
            number,
            Moving {
                client,
                request,
                deployment,
                to_host,
                items,
                failure: None,
            },
        );
        tracing::info!(storages = count, to_host, "moving a world's weights");
        for index in 0..count {
            if to_host {
                let bytes = self.moves.active[&number].items[index].bytes;
                let gpu = self.moves.active[&number].items[index].gpu;
                match crate::pages::shared_memory("uniserve-weights", bytes, None) {
                    Ok(descriptor) => {
                        let target = self.add_storage(
                            StorageRecord {
                                placement: Placement::Host { numa_node: None },
                                bytes,
                                segment: None,
                                fabric: None,
                            },
                            descriptor,
                            Some(gpu),
                        );
                        self.copy_item(number, index, target);
                    }
                    Err(error) => {
                        self.item_ended(number, index, Some(format!("host storage: {error}")));
                    }
                }
            } else {
                let Some(transfer_server) = self.transfers.client else {
                    self.item_ended(number, index, Some("no Transfer Server runs".to_owned()));
                    continue;
                };
                let allocation = self.moves.next_allocation;
                self.moves.next_allocation += 1;
                self.moves.allocating.insert(allocation, (number, index));
                let item = &self.moves.active[&number].items[index];
                self.events.push((
                    transfer_server,
                    Event::Allocate {
                        request: allocation,
                        gpu: item.gpu,
                        bytes: item.bytes,
                    },
                    Vec::new(),
                ));
            }
        }
    }

    /// The Transfer Server created the device storage of allocation
    /// `request` (`descriptor`, `bytes` long), or `failure` says why it did
    /// not: the restore's item copies into it.
    pub(crate) fn allocated(
        &mut self,
        request: u64,
        bytes: u64,
        failure: Option<&str>,
        descriptor: Option<OwnedFd>,
    ) {
        let Some((number, index)) = self.moves.allocating.remove(&request) else {
            return;
        };
        let Some(gpu) = self
            .moves
            .active
            .get(&number)
            .map(|moving| moving.items[index].gpu)
        else {
            return;
        };
        match (failure, descriptor) {
            (None, Some(descriptor)) => {
                let target = self.add_storage(
                    StorageRecord {
                        placement: Placement::Device { gpu },
                        bytes,
                        segment: None,
                        fabric: None,
                    },
                    descriptor,
                    None,
                );
                self.copy_item(number, index, target);
            }
            (failure, _) => {
                let reason = failure.unwrap_or("the device storage came without its descriptor");
                self.item_ended(number, index, Some(format!("device storage: {reason}")));
            }
        }
    }

    /// The copy of a move's item ended, as the Transfer Server reported
    /// (`end`).
    pub(crate) fn moved(&mut self, number: u64, index: usize, end: &TransferEnd) {
        let failure = match end {
            TransferEnd::Complete => None,
            TransferEnd::Cancelled => Some("the copy was cancelled".to_owned()),
            TransferEnd::Failed { message } => Some(message.clone()),
        };
        self.item_ended(number, index, failure);
    }

    /// The Transfer Server ended: the allocations it owed fail their moves.
    pub(crate) fn moves_lost_allocations(&mut self) {
        for (_, (number, index)) in std::mem::take(&mut self.moves.allocating) {
            self.item_ended(number, index, Some("the Transfer Server ended".to_owned()));
        }
    }

    /// A storage of `record` over `descriptor`, with no tensor yet: the
    /// copy a move fills; `moved_from` names the GPU of a host copy.
    fn add_storage(
        &mut self,
        record: StorageRecord,
        descriptor: OwnedFd,
        moved_from: Option<[u8; 16]>,
    ) -> StorageId {
        let id = self.next_storage;
        self.next_storage += 1;
        self.storages.insert(
            id,
            StorageEntry {
                record,
                _descriptor: descriptor,
                tensors: 0,
                moved_from,
                moving: true,
            },
        );
        id
    }

    /// Has the Transfer Server copy item `index` of move `number` into
    /// `target`.
    fn copy_item(&mut self, number: u64, index: usize, target: StorageId) {
        let Some(moving) = self.moves.active.get_mut(&number) else {
            return;
        };
        let item = &mut moving.items[index];
        item.target = Some(target);
        let (source, bytes) = (item.source, item.bytes);
        let deployment = moving.deployment.clone();
        let copied = self.copy_storage(&deployment, (source, target), bytes, (number, index));
        if let Err((_, message)) = copied {
            self.item_ended(number, index, Some(message));
        }
    }

    /// Item `index` of move `number` ended, with `failure` when it did not
    /// copy; the move ends with its last item.
    fn item_ended(&mut self, number: u64, index: usize, failure: Option<String>) {
        let Some(moving) = self.moves.active.get_mut(&number) else {
            return;
        };
        moving.items[index].ended = true;
        if let Some(failure) = failure {
            moving.failure.get_or_insert(failure);
        }
        if moving.items.iter().all(|item| item.ended) {
            self.end_move(number);
        }
    }

    /// Move `number` ended: its tensors lie in the copies and the storages
    /// they left are released, or, when a copy failed, every tensor stays
    /// and the copies are released. Answers the head.
    fn end_move(&mut self, number: u64) {
        let Some(moving) = self.moves.active.remove(&number) else {
            return;
        };
        let reply = match &moving.failure {
            Some(failure) => {
                for target in moving.items.iter().filter_map(|item| item.target) {
                    self.release_storage(target);
                }
                tracing::warn!(failure, to_host = moving.to_host, "a weight move failed");
                Reply::Error {
                    kind: ErrorKind::Resource,
                    message: format!("moving the weights failed: {failure}"),
                }
            }
            None => {
                let mut bytes = 0;
                for item in &moving.items {
                    let Some(target) = item.target else {
                        continue;
                    };
                    let count = self.relocate(item.source, target);
                    if let Some(entry) = self.storages.get_mut(&target) {
                        entry.tensors = count;
                        entry.moving = false;
                    }
                    self.release_storage(item.source);
                    // The world was retired while its weights moved: the
                    // copy holds nothing anyone retains.
                    if count == 0 {
                        self.release_storage(target);
                    } else {
                        // The Transfer Server mapped the copy for writing,
                        // and pinned a host copy: it unmaps it now, and a
                        // later task maps it as the read-only content it is.
                        self.allocation_released(Allocation::Storage { storage: target });
                    }
                    bytes += item.bytes;
                }
                tracing::info!(
                    storages = moving.items.len(),
                    bytes,
                    to_host = moving.to_host,
                    "moved a world's weights"
                );
                Reply::Moved {
                    storages: moving.items.len() as u64,
                    bytes,
                }
            }
        };
        self.answer(moving.client, moving.request, reply, &[]);
        if let Some(entry) = self.clients.get_mut(&moving.client) {
            entry.held = false;
        }
        self.serve_queue(moving.client);
    }

    /// The copies of tensors in storage `source` lie in `target` from now
    /// on, at its location: each is keyed anew by that location. A key that
    /// already has a copy there keeps that copy, which takes over this one's
    /// retainers. Returns how many copies lie in `target`; the caller
    /// releases `source`.
    fn relocate(&mut self, source: StorageId, target: StorageId) -> usize {
        let Some(location) = self
            .storages
            .get(&target)
            .map(|entry| entry.record.placement)
        else {
            return 0;
        };
        let moved: Vec<crate::TensorId> = self
            .tensors
            .iter()
            .filter(|(_, tensor)| tensor.storage == source)
            .map(|(id, _)| *id)
            .collect();
        let mut count = 0;
        for id in moved {
            let Some(mut tensor) = self.take_tensor(&id) else {
                continue;
            };
            match self.tensors.get_mut(&(id.0, location)) {
                Some(existing) => existing.retainers.append(&mut tensor.retainers),
                None => {
                    tensor.storage = target;
                    self.insert_tensor((id.0, location), tensor);
                    count += 1;
                }
            }
        }
        count
    }

    /// Releases storage `storage`, which no tensor lies in any more.
    fn release_storage(&mut self, storage: StorageId) {
        if self.storages.remove(&storage).is_some() {
            self.allocation_released(Allocation::Storage { storage });
        }
    }

    /// Refuses `bytes` of new device storage on `gpu` that would take
    /// `deployment`'s ledger of it past its quota, naming `what`.
    pub(crate) fn admit_device(
        &self,
        deployment: &str,
        gpu: [u8; 16],
        bytes: u64,
        what: &str,
    ) -> Result<(), (ErrorKind, String)> {
        for ledger in self.ledgers_of(&[gpu], Some(deployment)) {
            let (Some(quota), used) = (ledger.quota, ledger.used()) else {
                continue;
            };
            if used + bytes > quota {
                return Err((
                    ErrorKind::Quota,
                    format!(
                        "{bytes} bytes of {what} would take the storage ledger of GPU {} to {} \
                         bytes, past its quota of {quota} bytes:\n{}",
                        ::storage::protocol::gpu_name(&gpu),
                        used + bytes,
                        ledger.describe()
                    ),
                ));
            }
        }
        Ok(())
    }
}

/// The storages a move's copy reads and writes, as the Transfer Server maps
/// them: a device storage on its GPU, a host storage in host storage.
pub(crate) fn storage_of(record: &StorageRecord) -> LinkEnd {
    match record.placement {
        Placement::Device { gpu } => LinkEnd::Device { gpu },
        Placement::Host { .. } => LinkEnd::Host,
    }
}

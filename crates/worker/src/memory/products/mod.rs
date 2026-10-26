//! The rank's cross-call tensor store over Rust-owned storage.
//!
//! [`ProductStore`] owns every value a call publishes for later calls on this
//! rank or on other ranks (a *product*): its placement, its write state, its
//! read leases, its imports from other ranks and its exports to them.
//!
//! - **Placement.** A product with a scheduler buffer allocation is
//!   *persistent*: a range of the device's [`BufferPool`] arena, which is an
//!   exportable allocation so CUDA VMM exports stay in place. A scalar product
//!   without one lives in a *relay entry*: one element of a per-(device, dtype,
//!   field) arena of `(request_capacity + 1) * relay_depth` elements, indexed
//!   by request slot and depth index.
//! - **Writes.** Producers write the bound view (see
//!   [`ProductBuffer::view`]) and declare it produced, or hand a
//!   value that is copied natively into it. Writes never convert dtypes: a
//!   producer writes the store's dtype. A producer fence (an event on the
//!   explicit producer stream) orders consumers after the write.
//! - **Reads.** Consumers lease committed products; their consuming stream
//!   waits on the producer fence, and completing the lease records a reader
//!   fence on that stream.
//! - **Retirement.** Logical release is not reuse: a buffer's storage
//!   returns only after release, with no reader, every transfer and export
//!   successfully retired, and every fence complete. Failed or unknown
//!   completions keep the storage forever.
//!
//! [`State`] is the store's bookkeeping core, which [`ProductStore`]
//! serializes changes to.
//! Buffer and import handles share their state with read leases; only
//! physical completion (events, transfer and export retirement) decides when
//! storage can be reused.

mod export;
mod import;
mod product;
mod read;
mod relay;
mod scalar;
mod write;

pub use export::ProductExport;
pub use product::{Metadata, ProductBuffer, ValueRange};
pub use read::ProductRead;
pub use write::OutputPlacement;

// The store's parts share their items through this module.
use import::*;
use product::*;
use read::*;
use relay::*;
use write::*;

use crate::ffi::MemoryKind;
use crate::memory::allocation::StorageView;
use crate::memory::buffer::BufferPool;
use crate::stream::Issue;
use crate::transfer::exports::{self, Location};
use crate::{Error, EventPool, Result, lock};
use cuda::Event;
use ipc::CallId;
use ipc::{BufferId, RequestKey, TensorRef};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

/// The rank's cross-call tensor store.
pub struct ProductStore {
    max_feature_bytes: usize,
    devices: Vec<MemoryKind>,
    buffer_pool: Mutex<BufferPool>,
    events: Arc<EventPool>,
    /// Committed export locations per local product, revoked on release.
    exports: Mutex<HashMap<BufferId, Vec<Location>>>,
    state: Mutex<State>,
}

pub(crate) type BufferKey = (RequestKey, CallId, u16);

pub(crate) type CallKey = (RequestKey, CallId);

/// Rank-local values and bounded scalar relay entries: the store's buffers and
/// imports, the views of its arenas and of its relay entries.
struct State {
    byte_capacity: usize,
    request_capacity: usize,
    relay_depth: usize,
    buffers: HashMap<BufferKey, ProductBuffer>,
    imports: HashMap<BufferKey, ImportHandle>,
    arenas: HashMap<ArenaKey, StorageView>,
    views: HashMap<RelayKey, StorageView>,
    allocated_bytes: usize,
    calls: HashMap<CallKey, HashSet<BufferKey>>,
    entries: HashMap<EntryKey, Entry>,
    call_entries: HashMap<RelayCall, usize>,
}

impl State {
    pub(crate) fn new(byte_capacity: usize, request_capacity: usize, relay_depth: usize) -> Self {
        Self {
            byte_capacity,
            request_capacity,
            relay_depth,
            buffers: HashMap::new(),
            imports: HashMap::new(),
            arenas: HashMap::new(),
            views: HashMap::new(),
            allocated_bytes: 0,
            calls: HashMap::new(),
            entries: HashMap::new(),
            call_entries: HashMap::new(),
        }
    }
}

impl State {
    /// Check the whole allocation group before any view is created: no
    /// output repeats or is registered already. The scheduler's buffer
    /// placements bound persistent products; the store counts none.
    pub(crate) fn validate_bindings(&self, bindings: &[BufferId]) -> Result<()> {
        let mut seen = HashSet::new();
        for id in bindings {
            let key = buffer_key(*id);
            if !seen.insert(key) || self.buffers.contains_key(&key) {
                return Err(Error::Invalid(
                    "persistent output is already registered".into(),
                ));
            }
        }
        Ok(())
    }

    /// Capacity is committed only after the arena allocation succeeds.
    pub(crate) fn arena_capacity(&self, bytes: usize) -> Result<usize> {
        self.allocated_bytes
            .checked_add(bytes)
            .filter(|&projected| projected <= self.byte_capacity)
            .ok_or(Error::Resource("device-product byte capacity is exhausted"))
    }

    pub(crate) fn require_buffer(&self, buffer: &ProductBuffer) -> Result<()> {
        let value = lock(buffer);
        if value.backing.is_none()
            || !self
                .buffers
                .get(&buffer_key(value.id))
                .is_some_and(|current| std::ptr::eq(&**current, &**buffer))
        {
            return Err(Error::Invariant(
                "stale device-product physical generation".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn require_reference(&self, id: BufferId) -> Result<&ProductBuffer> {
        let buffer = self.buffers.get(&buffer_key(id))
            .filter(|buffer| lock(buffer).state == WriteState::Committed)
            .ok_or_else(|| Error::Invalid(format!(
                "unknown device-product reference: request {} produced by {:?} output {} generation {}",
                id.owner.request_id.0, id.producer_call_id, id.output_index, id.generation,
            )))?;
        if lock(buffer).id != id {
            return Err(Error::Invalid(
                "stale device-product logical generation".into(),
            ));
        }
        Ok(buffer)
    }

    pub(crate) fn commit_writes(&mut self, writes: &[ProductBuffer]) -> Result<()> {
        self.validate_writes(writes)?;
        let mut seen = HashSet::new();
        for write in writes {
            let value = lock(write);
            if value.state == WriteState::Deferred {
                continue;
            }
            if !seen.insert(buffer_key(value.id)) {
                return Err(Error::Invariant(
                    "device-product commit repeats an output".into(),
                ));
            }
        }

        for write in writes {
            let mut value = lock(write);
            if value.state == WriteState::Deferred {
                continue;
            }
            value.state = WriteState::Committed;
            self.calls
                .entry((value.id.owner, value.id.producer_call_id))
                .or_default()
                .insert(buffer_key(value.id));
        }
        Ok(())
    }

    pub(crate) fn validate_writes(&self, writes: &[ProductBuffer]) -> Result<()> {
        for write in writes {
            self.require_buffer(write)?;
            lock(write).require_candidate()?;
        }
        Ok(())
    }

    pub(crate) fn abandon_writes(&self, writes: &[ProductBuffer]) {
        for write in writes {
            if self.require_buffer(write).is_ok() {
                lock(write).released = true;
            }
        }
    }

    pub(crate) fn detach(&mut self, id: BufferId) {
        let call = (id.owner, id.producer_call_id);
        if let Some(buffers) = self.calls.get_mut(&call) {
            buffers.remove(&buffer_key(id));
            if buffers.is_empty() {
                self.calls.remove(&call);
            }
        }
    }

    pub(crate) fn release_calls(&mut self, calls: impl IntoIterator<Item = CallKey>) {
        for call in calls {
            if let Some(buffers) = self.calls.remove(&call) {
                for key in buffers {
                    if let Some(buffer) = self.buffers.get(&key) {
                        lock(buffer).released = true;
                    }
                }
            }
        }
    }

    pub(crate) fn release_buffers(&mut self, selected: &HashSet<BufferId>) -> Vec<BufferKey> {
        let keys = self
            .buffers
            .iter()
            .filter_map(|(&key, buffer)| selected.contains(&lock(buffer).id).then_some(key))
            .collect::<Vec<_>>();
        self.release_keys(&keys);
        keys
    }

    /// Retained features keep persistent storage across request completion;
    /// request-slot relay storage cannot outlive the slot's request.
    pub(crate) fn release_requests(
        &mut self,
        requests: &HashSet<RequestKey>,
        retained: &HashSet<BufferId>,
    ) -> Result<Vec<BufferKey>> {
        let mut keys = Vec::new();
        for (&key, buffer) in &self.buffers {
            let value = lock(buffer);
            if requests.contains(&value.id.owner) {
                if retained.contains(&value.id) {
                    if !matches!(value.backing, Some(Backing::Persistent(_))) {
                        return Err(Error::Invalid(
                            "finish cannot retain request-slot storage".into(),
                        ));
                    }
                } else {
                    keys.push(key);
                }
            }
        }
        self.release_keys(&keys);
        Ok(keys)
    }

    fn release_keys(&mut self, keys: &[BufferKey]) {
        for key in keys {
            if let Some(buffer) = self.buffers.get(key) {
                let id = lock(buffer).id;
                lock(buffer).released = true;
                self.detach(id);
            }
        }
    }

    /// Return retired owners and their backing only after the full query pass.
    /// The caller releases pool bindings and event references after this call.
    pub(crate) fn reclaim(&mut self) -> Result<Vec<(ProductBuffer, Backing)>> {
        let mut queried = HashMap::new();
        let mut ready = Vec::new();
        for (&key, buffer) in &self.buffers {
            if lock(buffer).access_retired(&mut queried)? {
                ready.push(key);
            }
        }

        let mut retired = Vec::with_capacity(ready.len());
        for key in ready {
            if let Some(buffer) = self.buffers.remove(&key) {
                let id = lock(&buffer).id;
                self.detach(id);
                let backing = self.release_backing(&buffer)?;
                retired.push((buffer, backing));
            }
        }
        Ok(retired)
    }

    pub(crate) fn release_backing(&mut self, buffer: &ProductBuffer) -> Result<Backing> {
        let backing = lock(buffer)
            .backing
            .take()
            .ok_or_else(|| Error::Invariant("tensor storage was retired more than once".into()))?;
        if let Backing::Relay((entry_key, dtype, field)) = &backing {
            let entry = self
                .entries
                .get_mut(entry_key)
                .ok_or_else(|| Error::Invariant("request-relay entry lost its owner".into()))?;
            entry.fields.remove(&(*dtype, *field));
            if entry.fields.is_empty() {
                self.call_entries
                    .remove(&(entry_key.0, entry_key.1, entry.call.0, entry.call.1));
                self.entries.remove(entry_key);
            }
        }
        Ok(backing)
    }

    pub(crate) fn relay_entry(
        &self,
        id: BufferId,
        device: MemoryKind,
        slot: usize,
    ) -> Result<usize> {
        let call = (device, slot, id.owner, id.producer_call_id);
        self.call_entries
            .get(&call)
            .copied()
            .or_else(|| {
                (0..self.relay_depth)
                    .find(|&index| !self.entries.contains_key(&(device, slot, index)))
            })
            .ok_or(Error::Resource(
                "request-relay unresolved window is exhausted",
            ))
    }

    pub(crate) fn bind_relay(&mut self, id: BufferId, relay: &RelayKey) -> Result<()> {
        let (entry_key, dtype, field) = relay;
        let entry = self.entries.entry(*entry_key).or_insert_with(|| Entry {
            call: (id.owner, id.producer_call_id),
            fields: HashSet::new(),
        });
        if !entry.fields.insert((*dtype, *field)) {
            return Err(Error::Invariant(
                "request-relay entry was assigned more than once".into(),
            ));
        }
        self.call_entries.insert(
            (entry_key.0, entry_key.1, id.owner, id.producer_call_id),
            entry_key.2,
        );
        Ok(())
    }

    /// After physical accesses drain, return owners for destruction outside
    /// the store lock. Existing reads retain their views.
    pub(crate) fn close(&mut self) -> Self {
        let empty = Self::new(self.byte_capacity, self.request_capacity, self.relay_depth);
        std::mem::replace(self, empty)
    }
}

impl ProductStore {
    /// A store of `byte_capacity` bytes of relay arenas, with features of at
    /// most `max_feature_bytes` on `devices`, `request_capacity` request slots
    /// of `relay_depth` relay entries each (both zero without relays).
    /// Persistent products lie in `buffer_pool`, whose scheduler placements
    /// bound them.
    pub fn new(
        byte_capacity: usize,
        max_feature_bytes: usize,
        devices: Vec<MemoryKind>,
        request_capacity: usize,
        relay_depth: usize,
        buffer_pool: BufferPool,
        events: Arc<EventPool>,
    ) -> Result<Self> {
        if max_feature_bytes == 0 || byte_capacity == 0 {
            return Err(Error::Invalid("tensor store capacities are invalid".into()));
        }
        if (request_capacity == 0) != (relay_depth == 0) {
            return Err(Error::Invalid(
                "request-relay dimensions must be complete".into(),
            ));
        }
        Ok(Self {
            max_feature_bytes,
            devices,
            buffer_pool: Mutex::new(buffer_pool),
            events,
            exports: Mutex::new(HashMap::new()),
            state: Mutex::new(State::new(byte_capacity, request_capacity, relay_depth)),
        })
    }

    /// Return retired buffers' pool ranges and event references.
    fn reclaim(&self, state: &mut State) -> Result<()> {
        let retired = state.reclaim()?;
        let mut released: HashMap<usize, (Arc<Event>, usize)> = HashMap::new();
        for (buffer, backing) in retired {
            if let Backing::Persistent(binding) = backing {
                lock(&self.buffer_pool).release(&binding)?;
            }
            for event in lock(&buffer).events() {
                released
                    .entry(Arc::as_ptr(event) as usize)
                    .or_insert_with(|| (Arc::clone(event), 0))
                    .1 += 1;
            }
        }
        for (event, count) in released.into_values() {
            self.events.release(&event, count)?;
        }
        Ok(())
    }

    fn release_backing(&self, state: &mut State, buffer: &ProductBuffer) -> Result<()> {
        if let Backing::Persistent(binding) = state.release_backing(buffer)? {
            lock(&self.buffer_pool).release(&binding)?;
        }
        Ok(())
    }

    /// A fence after the producer's writes on `stream`, retained `count`
    /// times (one reference per buffer it fences). Host values need none.
    fn producer_fence(
        &self,
        device: MemoryKind,
        stream: impl Issue,
        count: usize,
    ) -> Result<Option<Fence>> {
        let MemoryKind::Cuda(index) = device else {
            return Ok(None);
        };
        let producer = stream.stream().ok_or_else(|| {
            Error::Invalid("a device product write requires its producer stream".into())
        })?;
        if producer.device() != index {
            return Err(Error::Invalid(
                "a producer stream is on another device than its product".into(),
            ));
        }
        let event = self.events.fence(stream, false)?;
        if count > 1 {
            self.events.retain(&event, index, count - 1)?;
        }
        Ok(Some(Fence {
            event,
            stream: producer.raw(),
        }))
    }

    /// Order `stream` after the buffer's producer, once per fence.
    fn wait_producer(
        buffer: &Buffer,
        stream: impl Issue,
        waited: &mut HashSet<usize>,
    ) -> Result<()> {
        if let Some(fence) = &buffer.producer {
            let consumer = stream.stream().ok_or_else(|| {
                Error::Invalid("device product consumption requires its consumer stream".into())
            })?;
            if waited.insert(Arc::as_ptr(&fence.event) as usize) && consumer.raw() != fence.stream {
                stream.wait(&fence.event)?;
            }
        }
        Ok(())
    }

    /// Have the service pass again once the fences of a released buffer
    /// completed: the buffer's storage returns on that pass
    /// ([`Self::reclaim`]). The deferred work is nothing itself; the
    /// completion it waits for is the wake. `stream` is what records the
    /// fences that are not recorded yet.
    fn follow_fences(&self, buffer: &ProductBuffer, stream: impl Issue) -> Result<()> {
        let events: Vec<_> = lock(buffer).events().cloned().collect();
        for event in &events {
            // The deferral's own reference; the buffer keeps its own
            // until it is reclaimed.
            self.events.retain(event, self.events.device(event)?, 1)?;
        }
        stream.defer(&self.events, events, Box::new(|| Ok(())))
    }

    /// Follow the fences of a released buffer, through `stream`, and
    /// cancel the imports of a never-produced buffer.
    fn retire_buffer(&self, buffer: &ProductBuffer, stream: impl Issue) -> Result<()> {
        self.follow_fences(buffer, stream)?;
        let transfers: Vec<_> = {
            let value = lock(buffer);
            if value.state.produced() {
                Vec::new()
            } else {
                value
                    .transfers
                    .iter()
                    .map(|read| Arc::clone(read.read()))
                    .collect()
            }
        };
        for read in transfers {
            read.cancel()?;
        }
        Ok(())
    }

    fn require_reference(state: &State, reference: &TensorRef) -> Result<ProductBuffer> {
        let buffer = state.require_reference(reference.buffer_id())?.clone();
        if buffer.views().reference != *reference {
            return Err(Error::Invalid(
                "stale device-product logical generation".into(),
            ));
        }
        Ok(buffer)
    }

    /// Release the outputs of finished calls.
    pub fn release_calls(&self, calls: &[(RequestKey, CallId)]) {
        lock(&self.state).release_calls(calls.iter().copied());
    }

    /// Revoke the export locations registered for `buffers`: no reader
    /// claims them from now on, and each export retires once the readers
    /// that did have finished. The buffers' storage stays with its owner.
    pub(crate) fn revoke_buffer_exports<'a>(
        &self,
        buffers: impl IntoIterator<Item = &'a BufferId>,
    ) -> Result<()> {
        let revoked: Vec<_> = {
            let mut exports = lock(&self.exports);
            buffers
                .into_iter()
                .filter_map(|buffer| exports.remove(buffer))
                .flatten()
                .collect()
        };
        exports::release(&revoked)
    }

    /// Revoke the exports of released buffers and release the buffers.
    /// What follows a buffer's outstanding fences is deferred through
    /// `stream`.
    pub(crate) fn release_buffers(
        &self,
        buffers: &HashSet<BufferId>,
        stream: impl Issue,
    ) -> Result<()> {
        self.revoke_buffer_exports(buffers)?;
        let released: Vec<_> = {
            let mut state = lock(&self.state);
            let keys = state.release_buffers(buffers);
            keys.iter()
                .filter_map(|key| state.buffers.get(key).cloned())
                .collect()
        };
        for buffer in released {
            self.retire_buffer(&buffer, stream)?;
        }
        let mut state = lock(&self.state);
        self.reclaim(&mut state)
    }

    /// Release retired requests' products, except retained persistent ones,
    /// and revoke every export location registered for their buffers
    /// (products and published cache extents). What follows a product's
    /// outstanding fences is deferred through `stream`.
    pub fn release_requests(
        &self,
        requests: &HashSet<RequestKey>,
        retained: &HashSet<BufferId>,
        stream: impl Issue,
    ) -> Result<()> {
        let revoked: Vec<_> = {
            let mut exports = lock(&self.exports);
            let buffers: Vec<BufferId> = exports
                .keys()
                .copied()
                .filter(|buffer| requests.contains(&buffer.owner) && !retained.contains(buffer))
                .collect();
            buffers
                .iter()
                .filter_map(|buffer| exports.remove(buffer))
                .flatten()
                .collect()
        };
        exports::release(&revoked)?;
        let released: Vec<_> = {
            let mut state = lock(&self.state);
            let keys = state.release_requests(requests, retained)?;
            keys.iter()
                .filter_map(|key| state.buffers.get(key).cloned())
                .collect()
        };
        for buffer in released {
            self.retire_buffer(&buffer, stream)?;
        }
        let mut state = lock(&self.state);
        self.reclaim(&mut state)
    }

    /// Whether every selected buffer has retired. A read or export whose
    /// physical completion failed or is unknown is an error.
    pub fn retirement_ready(
        &self,
        buffers: &HashSet<BufferId>,
        requests: &HashSet<RequestKey>,
        retained: &HashSet<BufferId>,
    ) -> Result<bool> {
        let selected = |id: &BufferId| {
            buffers.contains(id) || (requests.contains(&id.owner) && !retained.contains(id))
        };
        let mut state = lock(&self.state);
        for buffer in state.buffers.values() {
            let value = lock(buffer);
            if selected(&value.id) {
                for read in &value.transfers {
                    read.read().retirement_ready()?;
                }
                for export in &value.exports {
                    if let Some(crate::Outcome::Failed(error)) = export.outcome() {
                        return Err(Error::transport_source(
                            "product export retirement failed",
                            error,
                        ));
                    }
                }
            }
        }
        self.reclaim(&mut state)?;
        Ok(!state
            .buffers
            .values()
            .any(|buffer| selected(&lock(buffer).id)))
    }

    /// Forget every product after physical accesses drained. Existing reads
    /// retain their views.
    pub(crate) fn close(&self) {
        lock(&self.exports).clear();
        let retired = lock(&self.state).close();
        drop(retired);
    }
}

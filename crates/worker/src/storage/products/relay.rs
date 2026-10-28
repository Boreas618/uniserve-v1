//! Relay entries: scalar products stored per request slot and depth index.

use super::*;
use crate::ffi::{DType, MemoryKind};
use crate::storage::allocation::{Storage, StorageView};
use crate::{Error, Result};
use ipc::CallId;
use ipc::{RequestKey, TensorRef};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// A relay entry: its storage, request slot and depth index.
pub(crate) type EntryKey = (MemoryKind, usize, usize);

/// A relay element: its entry, the field's dtype and index.
pub(crate) type RelayKey = (EntryKey, DType, usize);

/// A relay arena: its storage, dtype and field index.
pub(crate) type ArenaKey = (MemoryKind, DType, usize);

pub(super) type RelayCall = (MemoryKind, usize, RequestKey, CallId);

/// A call holds its relay entry until all scalar fields have retired.
pub(super) struct Entry {
    pub(super) call: CallKey,
    pub(super) fields: HashSet<(DType, usize)>,
}

impl ProductStore {
    pub(super) fn bind_relays(
        &self,
        state: &mut State,
        bindings: &[(TensorRef, MemoryKind)],
        request_slots: &HashMap<RequestKey, usize>,
    ) -> Result<Vec<ProductBuffer>> {
        if state.request_capacity == 0 || state.relay_depth == 0 {
            return Err(Error::Resource("worker has no request-relay arena"));
        }
        self.reclaim(state)?;
        let mut seen = HashSet::new();
        let mut fields = HashMap::<(RelayCall, DType), usize>::new();
        let mut buffers = Vec::new();
        let result = (|| -> Result<()> {
            for (reference, device) in bindings {
                let id = reference.buffer_id();
                let key = buffer_key(id);
                if !seen.insert(key) {
                    return Err(Error::Invalid(
                        "request-relay registration repeats an output identity".into(),
                    ));
                }
                if let Some(existing) = state.buffers.get(&key) {
                    let message = if lock(existing).state != WriteState::Committed {
                        "request-relay output already has a candidate"
                    } else if existing.views().reference != *reference {
                        "stale request-relay logical generation"
                    } else {
                        "request-relay output is already registered"
                    };
                    return Err(Error::Invalid(message.into()));
                }
                if id.generation == 0 {
                    return Err(Error::Invalid(
                        "request-relay registration requires a positive logical generation".into(),
                    ));
                }
                let slot = request_slots
                    .get(&reference.request_key)
                    .copied()
                    .unwrap_or(0);
                let shape = device_shape(reference);
                if slot == 0 || slot > state.request_capacity || shape.iter().any(|&dim| dim != 1) {
                    return Err(Error::Invalid(
                        "request-relay output has an invalid slot or scalar shape".into(),
                    ));
                }

                let device = *device;
                let dtype = product_dtype(reference.dtype)?;
                let call = (device, slot, id.owner, id.producer_call_id);
                let field = fields.entry((call, dtype)).or_default();
                let field_index = *field;
                *field += 1;
                let index = state.relay_entry(id, device, slot)?;
                let relay = ((device, slot, index), dtype, field_index);
                let view = match state.views.get(&relay) {
                    Some(view) => view.clone(),
                    None => {
                        let arena_key = (device, dtype, field_index);
                        if !state.arenas.contains_key(&arena_key) {
                            let count = state
                                .request_capacity
                                .checked_add(1)
                                .and_then(|rows| rows.checked_mul(state.relay_depth))
                                .ok_or(Error::Resource("request-relay byte capacity overflows"))?;
                            let bytes = count * dtype.bytes();
                            let projected = state.arena_capacity(bytes)?;
                            let storage = match device {
                                MemoryKind::Cuda(index) => Storage::device(index, bytes)?,
                                MemoryKind::Host | MemoryKind::PinnedHost => Storage::host(bytes),
                            };
                            let arena = StorageView::contiguous(
                                storage as Arc<dyn crate::ffi::Backing>,
                                0,
                                dtype,
                                &[count as i64],
                            )?;
                            state.arenas.insert(arena_key, arena);
                            state.allocated_bytes = projected;
                        }
                        let arena = state.arenas.get(&arena_key).ok_or_else(|| {
                            Error::Invariant("request-relay arena disappeared".into())
                        })?;
                        let view = arena.narrow(0, (slot * state.relay_depth + index) as i64, 1)?;
                        state.views.insert(relay, view.clone());
                        view
                    }
                };
                let buffer = create_buffer(
                    reference,
                    device,
                    view.clone(),
                    view,
                    vec![1],
                    None,
                    false,
                    Backing::Relay(relay),
                );
                state.bind_relay(id, &relay)?;
                state.buffers.insert(key, buffer.clone());
                buffers.push(buffer);
            }
            Ok(())
        })();
        if let Err(error) = result {
            self.rollback(state, buffers)?;
            return Err(error);
        }
        Ok(buffers)
    }
}

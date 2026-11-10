//! The resident token requests, and the requests whose policy view may have
//! moved since the policy was last told.
//!
//! A policy keeps its own state from the events it is told (`Updated` among
//! them), so the mechanism must tell it of every request whose view moved, and
//! a step must not walk the residents to find them. Every change of a
//! request's view starts from a mutable access to its state, an arrival, an
//! admission or one of its calls entering or leaving flight; each of those
//! marks the request here, and a step computes the views of the marked
//! requests only (`Scheduler::refresh_views`).

use std::collections::{BTreeSet, HashMap};

use crate::RequestId;
use crate::generation::RequestState;

/// The resident token requests by id, marking every request accessed mutably.
#[derive(Default)]
pub(crate) struct Residents {
    requests: HashMap<RequestId, RequestState>,
    /// The requests whose view may have moved since the last
    /// [`Self::take_changed`]: resident ones through this map's mutable
    /// accessors, and queued and media ones through [`Self::mark`].
    changed: BTreeSet<RequestId>,
}

impl Residents {
    pub(crate) fn get(&self, id: &RequestId) -> Option<&RequestState> {
        self.requests.get(id)
    }

    /// Mutable access to a resident's state, which marks its view as moved.
    pub(crate) fn get_mut(&mut self, id: &RequestId) -> Option<&mut RequestState> {
        let state = self.requests.get_mut(id)?;
        self.changed.insert(*id);
        Some(state)
    }

    pub(crate) fn insert(&mut self, id: RequestId, state: RequestState) -> Option<RequestState> {
        self.changed.insert(id);
        self.requests.insert(id, state)
    }

    pub(crate) fn remove(&mut self, id: &RequestId) -> Option<RequestState> {
        self.changed.insert(*id);
        self.requests.remove(id)
    }

    pub(crate) fn contains_key(&self, id: &RequestId) -> bool {
        self.requests.contains_key(id)
    }

    pub(crate) fn len(&self) -> usize {
        self.requests.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    pub(crate) fn keys(&self) -> impl Iterator<Item = &RequestId> {
        self.requests.keys()
    }

    pub(crate) fn values(&self) -> impl Iterator<Item = &RequestState> {
        self.requests.values()
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (&RequestId, &RequestState)> {
        self.requests.iter()
    }

    /// Mutable access to every resident, which marks every one of them.
    pub(crate) fn values_mut(&mut self) -> impl Iterator<Item = &mut RequestState> {
        self.changed.extend(self.requests.keys().copied());
        self.requests.values_mut()
    }

    /// Mutable access to every resident with its id, marking every one.
    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = (&RequestId, &mut RequestState)> {
        self.changed.extend(self.requests.keys().copied());
        self.requests.iter_mut()
    }

    /// Marks a request whose view moved without a mutable access to a
    /// resident's state: an arrival, a queue change, a media request's
    /// progress, or one of its calls entering or leaving flight.
    pub(crate) fn mark(&mut self, id: RequestId) {
        self.changed.insert(id);
    }

    /// The requests marked since the last call, in id order.
    pub(crate) fn take_changed(&mut self) -> BTreeSet<RequestId> {
        std::mem::take(&mut self.changed)
    }
}

impl<'a> IntoIterator for &'a Residents {
    type Item = (&'a RequestId, &'a RequestState);
    type IntoIter = std::collections::hash_map::Iter<'a, RequestId, RequestState>;

    fn into_iter(self) -> Self::IntoIter {
        self.requests.iter()
    }
}

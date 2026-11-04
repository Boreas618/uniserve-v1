//! The waiting queue: the requests that wait for admission, in the order the
//! policy admits them.
//!
//! Design §4.4: the mechanism admits from the head of the queue and stops at
//! the first request it cannot place, leaving it and the ones behind it
//! queued, so the order of the queue is the admission decision. `fcfs` queues
//! by arrival, `priority` by the request's own priority and then arrival, and
//! both use this structure with a different key.
//!
//! A policy learns of arrivals from [`Event::Arrived`](crate::Event::Arrived),
//! of admissions from [`Event::Admitted`](crate::Event::Admitted), and of the
//! end of a queued request from [`Event::Finished`](crate::Event::Finished); a
//! request stays queued until one of those says otherwise. `init` fills the
//! queue from the snapshot, so a policy put in mid-flight sees the requests
//! that already wait.

use crate::ids::RequestId;
use crate::view::RequestView;

/// One key a queue is ordered by.
///
/// Lower keys come first, as the deployment's priority convention has it (a
/// lower priority value is more urgent, and an earlier arrival comes first).
pub trait Order: Copy + Ord {
    /// The key of a request with `priority` that arrived at `arrival` (the
    /// mechanism's monotonic clock, in nanoseconds).
    fn of(priority: i32, arrival: u64) -> Self;
}

/// Arrival order: what `fcfs` admits by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ArrivalOrder(pub u64);

impl Order for ArrivalOrder {
    fn of(_priority: i32, arrival: u64) -> Self {
        Self(arrival)
    }
}

/// The request's own priority, then arrival: what `priority` admits by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PriorityOrder {
    /// The request's priority; lower is more urgent.
    pub priority: i32,
    /// The instant the request reached the main process.
    pub arrival: u64,
}

impl Order for PriorityOrder {
    fn of(priority: i32, arrival: u64) -> Self {
        Self { priority, arrival }
    }
}

/// A request waiting for admission, with the key its order puts it at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Queued<K> {
    key: K,
    request: RequestId,
}

/// The waiting requests, kept in the policy's order.
///
/// Insertion is ordered and idempotent per request: a request already queued
/// keeps its place, so a repeated arrival changes nothing. Removal scans the
/// queue, which is the same cost as the ordered insertion.
#[derive(Clone, Debug)]
pub struct Waiting<K: Order> {
    queued: Vec<Queued<K>>,
}

impl<K: Order> Default for Waiting<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Order> Waiting<K> {
    /// An empty queue.
    pub const fn new() -> Self {
        Self { queued: Vec::new() }
    }

    /// Queues a request, if it is not queued already.
    ///
    /// It goes after every request whose key is smaller or equal, so requests
    /// with equal keys keep the order they arrived in.
    pub fn push(&mut self, view: &RequestView) {
        if self.contains(view.id) {
            return;
        }
        let key = K::of(view.priority, view.arrived_at.0);
        let at = self.queued.partition_point(|queued| queued.key <= key);
        self.queued.insert(
            at,
            Queued {
                key,
                request: view.id,
            },
        );
    }

    /// Removes a request from wherever the order put it, returning whether it
    /// waited.
    pub fn remove(&mut self, request: RequestId) -> bool {
        match self
            .queued
            .iter()
            .position(|queued| queued.request == request)
        {
            Some(at) => {
                self.queued.remove(at);
                true
            }
            None => false,
        }
    }

    /// Whether a request waits.
    pub fn contains(&self, request: RequestId) -> bool {
        self.queued.iter().any(|queued| queued.request == request)
    }

    /// The queue in order: the admission decision.
    pub fn iter(&self) -> impl Iterator<Item = RequestId> + '_ {
        self.queued.iter().map(|queued| queued.request)
    }

    /// The request at the head of the queue.
    pub fn front(&self) -> Option<RequestId> {
        self.queued.first().map(|queued| queued.request)
    }

    /// How many requests wait.
    pub fn len(&self) -> usize {
        self.queued.len()
    }

    /// Whether nothing waits.
    pub fn is_empty(&self) -> bool {
        self.queued.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::PhaseId;
    use crate::time::Nanos;
    use crate::view::{RequestClass, RequestState};

    fn view(id: u64, priority: i32, arrived: u64) -> RequestView {
        RequestView {
            id: RequestId(id),
            state: RequestState::Queued,
            phase: None,
            priority,
            arrived_at: Nanos(arrived),
            admitted_at: None,
            prompt_tokens: 0,
            generated_tokens: 0,
            cached_prefix_tokens: 0,
            prefilled_tokens: 0,
            cached_pages: 0,
            last_emitted_at: None,
            class: RequestClass::Token,
            waiting_transfer: false,
            next_tokens: 1,
            reservation_units: 0,
            prefetch_tokens: 0,
            refill_tokens: None,
            remaining_tokens: 0,
            migratable: true,
            cache_rank: None,
        }
    }

    #[test]
    fn arrival_order_ignores_priority() {
        let mut waiting = Waiting::<ArrivalOrder>::new();
        waiting.push(&view(30, 9, 3));
        waiting.push(&view(10, 0, 1));
        waiting.push(&view(20, 5, 2));
        let order: Vec<u64> = waiting.iter().map(|id| id.0).collect();
        assert_eq!(order, vec![10, 20, 30]);
    }

    #[test]
    fn priority_order_breaks_ties_by_arrival() {
        let mut waiting = Waiting::<PriorityOrder>::new();
        waiting.push(&view(20, 5, 2));
        waiting.push(&view(10, 5, 1));
        waiting.push(&view(30, 1, 3));
        let order: Vec<u64> = waiting.iter().map(|id| id.0).collect();
        assert_eq!(order, vec![30, 10, 20]);
    }

    #[test]
    fn a_request_already_queued_keeps_its_place() {
        let mut waiting = Waiting::<ArrivalOrder>::new();
        waiting.push(&view(10, 0, 1));
        waiting.push(&view(20, 0, 2));
        waiting.push(&view(10, 7, 9));
        let order: Vec<u64> = waiting.iter().map(|id| id.0).collect();
        assert_eq!(order, vec![10, 20]);
        assert_eq!(waiting.len(), 2);
    }

    #[test]
    fn removal_says_whether_the_request_waited() {
        let mut waiting = Waiting::<ArrivalOrder>::new();
        waiting.push(&view(10, 0, 1));
        assert_eq!(waiting.front(), Some(RequestId(10)));
        assert!(waiting.remove(RequestId(10)));
        assert!(!waiting.remove(RequestId(10)));
        assert!(waiting.is_empty());
    }

    #[test]
    fn a_phase_of_the_request_is_opaque_to_the_queue() {
        let mut waiting = Waiting::<PriorityOrder>::new();
        let mut first = view(1, 0, 1);
        first.phase = Some(PhaseId(3));
        waiting.push(&first);
        assert_eq!(waiting.front(), Some(RequestId(1)));
    }
}

//! The waiting queue: the requests that wait for admission, in the order the
//! policy admits them.
//!
//! Design §4.4: the mechanism admits from the head of the queue and stops at
//! the first request it cannot place, leaving it and the ones behind it
//! queued, so the order of the queue is the admission decision. The queue is
//! ordered by the package's `order` argument ([`QueueOrder`]): by arrival, or
//! by the request's own priority and then arrival.
//!
//! A policy learns of arrivals from [`Event::Arrived`](scheduling::Event::Arrived),
//! of admissions from [`Event::Admitted`](scheduling::Event::Admitted), and of the
//! end of a waiting request from [`Event::Finished`](scheduling::Event::Finished); a
//! request stays queued until one of those says otherwise. `init` fills the
//! queue from the snapshot, so a policy put in mid-flight sees the requests
//! that already wait.

use scheduling::{QueueOrder, RequestId, RequestView};

/// A request waiting for admission, with the key the queue's order puts it at: lower keys come
/// first, as the deployment's priority convention has it (a lower priority value is more urgent,
/// and an earlier arrival comes first).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Entry {
    /// The request's priority under [`QueueOrder::Priority`], 0 under [`QueueOrder::Arrival`],
    /// then the instant it reached the main process (the mechanism's monotonic clock, in
    /// nanoseconds).
    key: (i32, u64),
    request: RequestId,
}

/// The waiting requests, kept in the policy's order.
///
/// Insertion is ordered and idempotent per request: a request already queued
/// keeps its place, so a repeated arrival changes nothing. Removal scans the
/// queue, which is the same cost as the ordered insertion.
#[derive(Clone, Debug)]
pub(crate) struct Waiting {
    order: QueueOrder,
    queued: Vec<Entry>,
}

impl Waiting {
    /// An empty queue that admits in `order`.
    pub(crate) const fn new(order: QueueOrder) -> Self {
        Self {
            order,
            queued: Vec::new(),
        }
    }

    /// The key `view` waits at.
    fn key(&self, view: &RequestView) -> (i32, u64) {
        let priority = match self.order {
            QueueOrder::Arrival => 0,
            QueueOrder::Priority => view.priority,
        };
        (priority, view.arrived_at.0)
    }

    /// Queues a request, if it is not queued already.
    ///
    /// It goes after every request whose key is smaller or equal, so requests
    /// with equal keys keep the order they arrived in.
    pub(crate) fn push(&mut self, view: &RequestView) {
        if self.contains(view.id) {
            return;
        }
        let key = self.key(view);
        let at = self.queued.partition_point(|queued| queued.key <= key);
        self.queued.insert(
            at,
            Entry {
                key,
                request: view.id,
            },
        );
    }

    /// Removes a request from wherever the order put it, returning whether it
    /// waited.
    pub(crate) fn remove(&mut self, request: RequestId) -> bool {
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
    pub(crate) fn contains(&self, request: RequestId) -> bool {
        self.queued.iter().any(|queued| queued.request == request)
    }

    /// The queue in order: the admission decision.
    pub(crate) fn iter(&self) -> impl Iterator<Item = RequestId> + '_ {
        self.queued.iter().map(|queued| queued.request)
    }
}

#[cfg(test)]
mod tests {
    use scheduling::{Nanos, PhaseId, RequestClass, RequestStatus};

    use super::*;

    fn view(id: u64, priority: i32, arrived: u64) -> RequestView {
        RequestView {
            id: RequestId(id),
            state: RequestStatus::Waiting,
            phase: None,
            priority,
            arrived_at: Nanos(arrived),
            admitted_at: None,
            prompt_tokens: 0,
            prefill_tokens: 0,
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

    fn order(waiting: &Waiting) -> Vec<u64> {
        waiting.iter().map(|id| id.0).collect()
    }

    #[test]
    fn arrival_order_ignores_priority() {
        let mut waiting = Waiting::new(QueueOrder::Arrival);
        waiting.push(&view(30, 9, 3));
        waiting.push(&view(10, 0, 1));
        waiting.push(&view(20, 5, 2));
        assert_eq!(order(&waiting), vec![10, 20, 30]);
    }

    #[test]
    fn priority_order_breaks_ties_by_arrival() {
        let mut waiting = Waiting::new(QueueOrder::Priority);
        waiting.push(&view(20, 5, 2));
        waiting.push(&view(10, 5, 1));
        waiting.push(&view(30, 1, 3));
        assert_eq!(order(&waiting), vec![30, 10, 20]);
    }

    #[test]
    fn a_request_already_queued_keeps_its_place() {
        let mut waiting = Waiting::new(QueueOrder::Arrival);
        waiting.push(&view(10, 0, 1));
        waiting.push(&view(20, 0, 2));
        waiting.push(&view(10, 7, 9));
        assert_eq!(order(&waiting), vec![10, 20]);
    }

    #[test]
    fn removal_says_whether_the_request_waited() {
        let mut waiting = Waiting::new(QueueOrder::Arrival);
        waiting.push(&view(10, 0, 1));
        assert_eq!(order(&waiting), vec![10]);
        assert!(waiting.remove(RequestId(10)));
        assert!(!waiting.remove(RequestId(10)));
        assert!(order(&waiting).is_empty());
    }

    #[test]
    fn a_phase_of_the_request_is_opaque_to_the_queue() {
        let mut waiting = Waiting::new(QueueOrder::Priority);
        let mut first = view(1, 0, 1);
        first.phase = Some(PhaseId(3));
        waiting.push(&first);
        assert_eq!(order(&waiting), vec![1]);
    }
}

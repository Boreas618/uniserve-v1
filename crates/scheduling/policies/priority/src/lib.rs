//! The `priority` scheduling policy: the request's own priority first, then
//! arrival.
//!
//! Design §4.4: "`priority` 在接纳与装批时先比请求自带的 priority" — at
//! admission and at batch assembly, a request that carries a more urgent
//! priority goes first, and requests of equal priority keep the order they
//! arrived in. Everything else is `fcfs`, so the body is the guest's shared
//! [`scheduling_guest::Arrival`] and the only difference is the ordering key.
//!
//! Today the same policy exists as an ordering of the waiting queue and of
//! the resident set: `scheduler/admission.rs` inserts a request after every
//! queued request with a lower priority value or an equal one and an earlier
//! arrival, and `Scheduler::reconfigure` re-sorts the queue the same way when
//! the policy changes. The batch assembly order follows admission order
//! within a phase class, so the priority decides there too.
//!
//! The priority is the request's own parameter and scheduling treats it as
//! opaque: only its order matters.

use scheduling_guest::{Arrival, PriorityOrder, export_policy};

/// The `priority` policy: the shared arrival-order body, ordered by the
/// request's priority and then by arrival.
pub type Priority = Arrival<PriorityOrder>;

export_policy!(Priority);

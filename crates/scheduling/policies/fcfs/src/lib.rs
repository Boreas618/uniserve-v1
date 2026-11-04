//! The `fcfs` scheduling policy: first come, first served.
//!
//! Design §4.4 makes this the default policy and requires it to reproduce the
//! rules today's scheduler applies, item by item. Its order is arrival: a
//! request's place in the waiting queue is the instant it reached the main
//! process, and the batch is assembled in admission order within a phase
//! class.
//!
//! The body of the policy is [`scheduling_guest::Arrival`], which the guest
//! crate holds because `priority` and every later arrival-order policy differ
//! from this one only in the key they order by (design §4.4: "`fcfs` 与
//! `priority` 都建在它上面，新的调度策略只写不同的部分"). The mapping from each
//! decision of the design's table to the code is the table on
//! [`scheduling_guest::Arrival`]; what is specific to `fcfs` is one line here:
//! the key is [`ArrivalOrder`].
//!
//! # Why arrival alone
//!
//! Today's waiting queue is a `VecDeque` appended to in submission order
//! (`scheduler/admission.rs`, `enqueue`), and admission examines only its
//! head, so the order in which requests are admitted is exactly the order in
//! which they arrived. Batch assembly then takes the residents by assembly
//! class and, within a class, by the order they were admitted
//! (`batching.rs`, `assembly_order`), which is arrival order again because
//! admission is. Neither decision reads the request's own `priority` under
//! this policy, so the ordering key is arrival alone.

use scheduling_guest::{Arrival, ArrivalOrder, export_policy};

/// The `fcfs` policy: the shared arrival-order body, ordered by arrival.
pub type Fcfs = Arrival<ArrivalOrder>;

export_policy!(Fcfs);

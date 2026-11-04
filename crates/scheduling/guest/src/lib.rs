//! The interface between the scheduling mechanism and a scheduling policy.
//!
//! This crate is compiled into every policy module (wasm32) and into the
//! mechanism (native), so both sides are built from the same types and the same
//! wire format. It holds, as design §4.4 gives them, the policy interface, the
//! events the mechanism hands to the policy, the decisions the policy returns,
//! and the snapshot and views a policy reads.
//!
//! # Append-only views and events
//!
//! Views and events are append-only lists (design §4.4): the mechanism is the
//! epoch layer and a policy is the program layer, so a newer mechanism may run
//! an older policy module. The wire format is therefore self-describing JSON:
//! a reader ignores struct fields it does not know and skips an event kind it
//! does not know ([`event::Event::Unknown`]), and every field a policy writes
//! has a default so a reader tolerates a field an older policy did not write.
//! Adding a field, an event kind, a view or a decision entry needs no ABI
//! change; changing what an existing field means does.
//!
//! # The ABI
//!
//! A policy module exports the functions named in [`wire`]: `sched_alloc` and
//! `sched_free` for the linear-memory buffers, and one function per decision
//! with its input as `(ptr, len)` and its reply packed into a `u64`
//! (`ptr << 32 | len`, `0` for an empty reply) or, for [`Call::Route`] and
//! [`Call::Place`], the chosen index directly. The mechanism resets the fuel
//! budget before every call and treats a trap or an exhausted budget as a
//! policy failure (design §4.4).
//!
//! A policy implements [`policy::Policy`]; the `export_policy!` macro (in the
//! guest helpers) generates the exports for it.

pub mod abi;
pub mod arrival;
pub mod budget;
pub mod decision;
pub mod event;
pub mod ids;
pub mod limits;
pub mod lru;
pub mod phases;
pub mod policy;
pub mod time;
pub mod view;
pub mod waiting;
pub mod wire;

pub use abi::*;
pub use arrival::Arrival;
pub use budget::{Shares, Step, prefill_chunk, work_tokens};
pub use decision::{Inflight, Plan, Room, Shortage, Storage, Victim, Warming, Work};
pub use event::{End, Event};
pub use ids::{ComponentId, EncodingId, PageId, PhaseId, RequestId};
pub use limits::{Limits, Parameters};
pub use lru::Lru;
pub use phases::Phases;
pub use policy::{Instantiation, Policy};
pub use time::{Duration, Nanos};
pub use view::{
    CallView, EncodingView, ExpertLoad, ExpertOrder, PageView, PhaseClass, PhaseKind, PhaseView,
    RankView, ReplicaView, RequestClass, RequestState, RequestView, Snapshot,
};
pub use waiting::{ArrivalOrder, Order, PriorityOrder, Waiting};

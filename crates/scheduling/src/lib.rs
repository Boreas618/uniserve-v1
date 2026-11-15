//! The interface between the scheduling mechanism and a scheduling policy.
//!
//! This crate is compiled into every policy module (wasm32) and into the
//! mechanism (native), so both sides are built from the same types and the same
//! wire format. It holds, as design §4.4 gives them, the policy interface, the
//! events the mechanism hands to the policy, the decisions the policy returns,
//! the snapshot and views a policy reads, and the parameters and limits of a
//! policy package.
//!
//! # One version per release
//!
//! The mechanism and a policy speak one version of this interface,
//! [`INTERFACE_VERSION`]: a module states the version it was compiled against
//! (`sched_interface_version`), and the mechanism refuses to instantiate a
//! module of another. Every message is decoded as exactly the type this
//! release declares, so a kind of event, a field or a value of an enum that
//! the release does not have is refused rather than skipped.
//!
//! # The ABI
//!
//! A policy module exports the functions named in [`wire`]: `sched_alloc` and
//! `sched_free` for the linear-memory buffers, and one function per decision
//! with its input as `(ptr, len)` and its reply packed into a `u64`
//! (`ptr << 32 | len`, `0` for an empty reply) or, for [`wire::Call::Route`] and
//! [`wire::Call::Place`], the chosen index directly. The mechanism resets the fuel
//! budget before every call and treats a trap or an exhausted budget as a
//! policy failure (design §4.4).
//!
//! A policy implements [`policy::Policy`] and is constructed from its
//! package's [`Parameters`] (`TryFrom<Parameters>`); the [`export_policy!`]
//! macro generates the exports for it.

/// The version of the scheduling policy interface: the calls of [`wire`] and
/// what their messages mean. Every change to a call, a message or what one of
/// its fields means raises it.
pub const INTERFACE_VERSION: u32 = 5;

pub mod abi;
pub mod decision;
pub mod event;
pub mod ids;
pub mod limits;
pub mod policy;
pub mod time;
pub mod view;
pub mod wire;

pub use abi::*;
pub use decision::{DispatchOutput, Inflight, Room, Shortage, Storage, Victim, Work};
pub use event::{End, Event};
pub use ids::{EncodingId, PageId, PhaseId, RequestId, StageId};
pub use limits::{Limits, Parameters, QueueOrder, Unstated, stated};
pub use policy::{Instantiation, Policy};
pub use time::{Duration, Nanos};
pub use view::{
    CallView, EncodingView, PageView, Phase, PhaseClass, PhaseView, RankView, ReplicaView,
    RequestClass, RequestStatus, RequestView, Snapshot,
};

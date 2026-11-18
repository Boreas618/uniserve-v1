//! The interface between the deployment and its programs: the program SDK.
//!
//! A policy package's programs are crates compiled to wasm32 modules, each run by a host in its
//! own instance. This crate is compiled into every program and into every host, so both sides
//! are built from the same types and the same wire format.
//!
//! # Programs
//!
//! Every program implements [`program::Program`] (`new`, `step`) and exports itself with
//! [`export!`]; [`program`] states the exports, the host functions and the call convention, and
//! [`prelude`] is what a program crate starts from. The inputs and outputs of the programs are
//! [`control`] (the leader daemon's control programs: [`control::State`] in, a configuration
//! out), [`routing`] (the leader daemon's routing program: [`routing::Routing`] in,
//! [`routing::Route`] out) and [`kv`] (the Storage Servers' KV program: handoff paths and
//! tiering moves).
//!
//! # The scheduling policy
//!
//! A scheduler's program is the scheduling policy, which keeps its own interface: the policy
//! interface, the events the mechanism hands to the policy, the decisions the policy returns,
//! the snapshot and views a policy reads, and the parameters and limits of a policy package
//! (design §4.4).
//!
//! The mechanism and a policy speak one version of this interface,
//! [`INTERFACE_VERSION`]: a module states the version it was compiled against
//! (`sched_interface_version`), and the mechanism refuses to instantiate a
//! module of another. Every message is decoded as exactly the type this
//! release declares, so a kind of event, a field or a value of an enum that
//! the release does not have is refused rather than skipped.
//!
//! A policy module exports the functions named in [`wire`]: `sched_alloc` and
//! `sched_free` for the linear-memory buffers, and one function per decision
//! with its input as `(ptr, len)` and its reply packed into a `u64`
//! (`ptr << 32 | len`, `0` for an empty reply) or, for [`wire::Call::Place`], the
//! chosen index directly. The mechanism resets the fuel
//! budget before every call and treats a trap or an exhausted budget as a
//! policy failure (design §4.4).
//!
//! A policy implements [`policy::Policy`] and is constructed from its
//! package's [`Parameters`] (`TryFrom<Parameters>`); the [`export_policy!`]
//! macro generates the exports for it.

/// The version of the scheduling policy interface: the calls of [`wire`] and
/// what their messages mean. Every change to a call, a message or what one of
/// its fields means raises it.
pub const INTERFACE_VERSION: u32 = 7;

pub mod abi;
pub mod control;
pub mod decision;
pub mod event;
pub mod ids;
pub mod kv;
pub mod limits;
pub mod policy;
pub mod prelude;
pub mod program;
pub mod routing;
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
    CallView, EncodingView, PageView, Phase, PhaseClass, PhaseView, ReplicaView, RequestClass,
    RequestStatus, RequestView, Snapshot,
};

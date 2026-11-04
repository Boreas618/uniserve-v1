//! The wire between the mechanism and a policy module.
//!
//! Both sides are built from this crate, so they agree on the encoding: every
//! message is JSON of the corresponding type (see the crate documentation for
//! why JSON and not a compact binary format), and the buffers of the module's
//! linear memory carry it: the mechanism writes a message with `sched_alloc`
//! and reads a reply through the packed pointer and length a call returns.
//!
//! # Call convention
//!
//! - `sched_alloc(len: u32) -> u32`: a buffer of `len` bytes the mechanism
//!   writes a message into.
//! - `sched_free(ptr: u32, len: u32)`: releases a buffer this module allocated,
//!   whether the mechanism wrote it or the module returned it.
//! - `sched_init(ptr, len) -> u64`, `sched_observe(ptr, len) -> u64`,
//!   `sched_dispatch(ptr, len) -> u64`, `sched_reclaim(ptr, len) -> u64`,
//!   `sched_warm(ptr, len) -> u64`, `sched_experts(ptr, len) -> u64`:
//!   the reply is `ptr << 32 | len` of a buffer the module allocated and the
//!   mechanism frees with `sched_free`, or `0` for an empty reply.
//! - `sched_route(ptr, len) -> u32`, `sched_place(ptr, len) -> u32`: the
//!   chosen index directly.
//! - `sched_limits() -> u64`: the reply buffer.
//!
//! A trap, including an exhausted fuel budget, is a policy failure. A reply
//! pointer or length outside the module's memory, or bytes that do not decode
//! to the expected type, are a policy failure of the `Malformed` kind.

use serde::Serialize;
use serde::de::DeserializeOwned;

/// The export that allocates a message buffer.
pub const EXPORT_ALLOC: &str = "sched_alloc";
/// The export that releases a buffer.
pub const EXPORT_FREE: &str = "sched_free";
/// The export that instantiates: its message is [`crate::policy::Instantiation`].
pub const EXPORT_INIT: &str = "sched_init";
/// The export that observes events: its message is `Vec<Event>`.
pub const EXPORT_OBSERVE: &str = "sched_observe";
/// The export that dispatches: its message is [`crate::decision::Room`].
pub const EXPORT_DISPATCH: &str = "sched_dispatch";
/// The export that reclaims: its message is [`crate::decision::Shortage`].
pub const EXPORT_RECLAIM: &str = "sched_reclaim";
/// The export that routes: its message is `(&RequestView, &[RankView])` as a
/// two-element tuple.
pub const EXPORT_ROUTE: &str = "sched_route";
/// The export that places: its message is `(&CallView, &[ReplicaView])`.
pub const EXPORT_PLACE: &str = "sched_place";
/// The export that warms: its message is [`crate::decision::Room`].
pub const EXPORT_WARM: &str = "sched_warm";
/// The export that places experts: its message is [`crate::view::ExpertLoad`].
pub const EXPORT_EXPERTS: &str = "sched_experts";
/// The export that reads the limits, with no message.
pub const EXPORT_LIMITS: &str = "sched_limits";

/// One call into a policy module. The names are the design's names for the
/// decisions (design §4.4); the mechanism reports a failure by them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Call {
    Init,
    Observe,
    Dispatch,
    Reclaim,
    Route,
    Place,
    Warm,
    Experts,
    Limits,
}

impl Call {
    /// Every call, in the order a policy's life makes them.
    pub const ALL: [Call; 9] = [
        Call::Init,
        Call::Observe,
        Call::Dispatch,
        Call::Reclaim,
        Call::Route,
        Call::Place,
        Call::Warm,
        Call::Experts,
        Call::Limits,
    ];

    /// The export this call drives.
    pub fn export(self) -> &'static str {
        match self {
            Call::Init => EXPORT_INIT,
            Call::Observe => EXPORT_OBSERVE,
            Call::Dispatch => EXPORT_DISPATCH,
            Call::Reclaim => EXPORT_RECLAIM,
            Call::Route => EXPORT_ROUTE,
            Call::Place => EXPORT_PLACE,
            Call::Warm => EXPORT_WARM,
            Call::Experts => EXPORT_EXPERTS,
            Call::Limits => EXPORT_LIMITS,
        }
    }

    /// The decision's name in design §4.4, used in failure reports.
    pub fn method(self) -> &'static str {
        match self {
            Call::Init => "init",
            Call::Observe => "observe",
            Call::Dispatch => "dispatch",
            Call::Reclaim => "reclaim",
            Call::Route => "route",
            Call::Place => "place",
            Call::Warm => "warm",
            Call::Experts => "experts",
            Call::Limits => "limits",
        }
    }

    /// Whether the reply is the chosen index itself, not a buffer.
    pub fn replies_with_index(self) -> bool {
        matches!(self, Call::Route | Call::Place)
    }

    /// Whether the call takes a message. Only `limits` takes none.
    pub fn takes_message(self) -> bool {
        !matches!(self, Call::Limits)
    }
}

/// Packs a reply buffer's pointer and length.
pub fn pack(ptr: u32, len: u32) -> u64 {
    (u64::from(ptr) << 32) | u64::from(len)
}

/// Splits a packed reply; `None` is the empty reply.
pub fn unpack(reply: u64) -> Option<(u32, u32)> {
    if reply == 0 {
        return None;
    }
    Some(((reply >> 32) as u32, reply as u32))
}

/// Encodes a message for the module's linear memory.
pub fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>, serde_json::Error> {
    serde_json::to_vec(value)
}

/// Decodes a message from the module's linear memory.
pub fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, serde_json::Error> {
    serde_json::from_slice(bytes)
}

//! The guest side of the call convention: the buffers of the module's linear
//! memory, and the exports a policy module offers the mechanism.
//!
//! [`crate::wire`] names the exports and the message of each; this module
//! implements them once, for every policy, so a policy crate writes only its
//! `Policy` implementation and invokes [`export_policy!`].
//!
//! # State
//!
//! One policy instance serves one scheduler thread, and wasmtime gives each
//! instance its own linear memory, so the policy lives in a module-level slot
//! that [`sched_init`] fills. wasm32 without threads runs one export at a
//! time, and the mechanism calls `init` once before any other export, which is
//! what makes the single mutable slot sound. A second instance in the same
//! process (the route instance on the control thread, or a trial instance
//! during precopy) is a second wasmtime instance with its own memory, so the
//! slots are unrelated.
//!
//! # Failure
//!
//! A policy never reports a bad decision here: the mechanism validates every
//! decision and owns the hard limits (design §4.4). A trap is the policy's
//! only failure — an exhausted fuel budget, or bytes that do not decode to the
//! message the export expects, which the mechanism reports as a malformed
//! reply of that call.

use crate::decision::{Plan, Room, Shortage, Victim};
use crate::event::Event;
use crate::limits::Limits;
use crate::policy::{Init, Policy};
use crate::view::{CallView, ExpertLoad, ExpertOrder, RankView, ReplicaView, RequestView};
use crate::wire::{decode, encode, pack};

/// A buffer of `len` bytes in this module's memory, which the mechanism
/// writes a message into.
///
/// The allocation is exactly `len` bytes, so [`sched_free`] can rebuild it
/// from the pointer and the same length.
#[unsafe(no_mangle)]
pub extern "C" fn sched_alloc(len: u32) -> u32 {
    let len = len as usize;
    let buffer = vec![0u8; len].into_boxed_slice();
    u32::try_from(Box::into_raw(buffer) as *mut u8 as usize)
        .expect("a wasm32 pointer fits in a u32")
}

/// Releases the buffer of `len` bytes at `ptr`, whether the mechanism wrote it
/// or this module returned it as a reply.
///
/// # Safety
///
/// `ptr` must come from [`sched_alloc`] or from a reply buffer this module
/// packed, with the length it was allocated or packed with.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sched_free(ptr: u32, len: u32) {
    if ptr == 0 {
        return;
    }
    let len = len as usize;
    let raw = ptr as *mut u8;
    // SAFETY: the caller hands back a buffer this module allocated with this
    // exact length (see `sched_alloc` and `reply`).
    drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(raw, len)) });
}

/// The message the mechanism wrote into this module's memory.
///
/// # Safety
///
/// `ptr` and `len` must name a buffer the mechanism filled with the message of
/// the export being called, and must not be used after that call returns.
pub unsafe fn message<'a>(ptr: u32, len: u32) -> &'a [u8] {
    // SAFETY: the mechanism owns the buffer until the call returns.
    unsafe { std::slice::from_raw_parts(ptr as *const u8, len as usize) }
}

/// Decodes an export's message, a trap when the bytes are not it.
fn read<T: serde::de::DeserializeOwned>(bytes: &[u8], what: &str) -> T {
    decode(bytes).unwrap_or_else(|error| panic!("{what} message does not decode: {error}"))
}

/// Copies `reply` into a fresh buffer and packs its pointer and length, which
/// the mechanism frees with [`sched_free`]. A reply that fails to encode is a
/// trap: it is a defect in the policy, not a decision.
fn reply<T: serde::Serialize>(reply: &T, what: &str) -> u64 {
    let bytes = encode(reply).unwrap_or_else(|error| panic!("{what} reply does not encode: {error}"));
    if bytes.is_empty() {
        return 0;
    }
    let len = u32::try_from(bytes.len()).expect("a reply fits in a u32 length");
    let ptr = sched_alloc(len);
    // SAFETY: the buffer is exactly `len` bytes, freshly allocated, and
    // nothing else holds it.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, len as usize);
    }
    pack(ptr, len)
}

/// Generates the exports of the `Policy` implementation `$policy`.
///
/// A policy crate invokes this once: the module's only state is the instance
/// it names.
#[macro_export]
macro_rules! export_policy {
    ($policy:ty) => {
        /// The policy of this module. `sched_init` fills it; see the module
        /// documentation for why one mutable slot is sound.
        static mut POLICY: ::core::option::Option<$policy> = ::core::option::Option::None;

        /// Runs `body` with the instantiated policy, a trap before `init`.
        fn policy<R>(body: impl FnOnce(&mut $policy) -> R) -> R {
            // SAFETY: the mechanism calls `sched_init` once before any other
            // export, wasm32 without threads runs one export at a time, and
            // nothing else touches the slot.
            unsafe {
                let slot = &raw mut POLICY;
                let policy = (*slot)
                    .as_mut()
                    .unwrap_or_else(|| panic!("sched_init runs before every other export"));
                body(policy)
            }
        }

        #[unsafe(no_mangle)]
        extern "C" fn sched_init(ptr: u32, len: u32) -> u64 {
            // SAFETY: the mechanism's message buffer lives for this call.
            let bytes = unsafe { $crate::abi::message(ptr, len) };
            let init: $crate::Init = $crate::abi::init_message(bytes);
            let mut policy = <$policy as ::core::default::Default>::default();
            <$policy as $crate::Policy>::init(&mut policy, &init);
            // SAFETY: as in `policy`; `init` is called once.
            unsafe {
                let slot = &raw mut POLICY;
                *slot = ::core::option::Option::Some(policy);
            }
            0
        }

        #[unsafe(no_mangle)]
        extern "C" fn sched_observe(ptr: u32, len: u32) -> u64 {
            // SAFETY: the mechanism's message buffer lives for this call.
            let bytes = unsafe { $crate::abi::message(ptr, len) };
            let events: ::std::vec::Vec<$crate::Event> =
                $crate::abi::events_message(bytes);
            policy(|policy| <$policy as $crate::Policy>::observe(policy, &events));
            0
        }

        #[unsafe(no_mangle)]
        extern "C" fn sched_dispatch(ptr: u32, len: u32) -> u64 {
            // SAFETY: the mechanism's message buffer lives for this call.
            let bytes = unsafe { $crate::abi::message(ptr, len) };
            let room: $crate::Room = $crate::abi::room_message(bytes);
            let plan: $crate::Plan = policy(|policy| <$policy as $crate::Policy>::dispatch(policy, &room));
            $crate::abi::dispatch_reply(&plan)
        }

        #[unsafe(no_mangle)]
        extern "C" fn sched_reclaim(ptr: u32, len: u32) -> u64 {
            // SAFETY: the mechanism's message buffer lives for this call.
            let bytes = unsafe { $crate::abi::message(ptr, len) };
            let shortage: $crate::Shortage = $crate::abi::shortage_message(bytes);
            let victims: ::std::vec::Vec<$crate::Victim> =
                policy(|policy| <$policy as $crate::Policy>::reclaim(policy, &shortage));
            $crate::abi::victims_reply(&victims)
        }

        #[unsafe(no_mangle)]
        extern "C" fn sched_route(ptr: u32, len: u32) -> u32 {
            // SAFETY: the mechanism's message buffer lives for this call.
            let bytes = unsafe { $crate::abi::message(ptr, len) };
            let (request, ranks) = $crate::abi::route_message(bytes);
            policy(|policy| <$policy as $crate::Policy>::route(policy, &request, &ranks))
                as u32
        }

        #[unsafe(no_mangle)]
        extern "C" fn sched_place(ptr: u32, len: u32) -> u32 {
            // SAFETY: the mechanism's message buffer lives for this call.
            let bytes = unsafe { $crate::abi::message(ptr, len) };
            let (call, replicas) = $crate::abi::place_message(bytes);
            policy(|policy| <$policy as $crate::Policy>::place(policy, &call, &replicas))
                as u32
        }

        #[unsafe(no_mangle)]
        extern "C" fn sched_warm(ptr: u32, len: u32) -> u64 {
            // SAFETY: the mechanism's message buffer lives for this call.
            let bytes = unsafe { $crate::abi::message(ptr, len) };
            let room: $crate::Room = $crate::abi::room_message(bytes);
            let pages: ::std::vec::Vec<$crate::PageId> =
                policy(|policy| <$policy as $crate::Policy>::warm(policy, &room));
            $crate::abi::pages_reply(&pages)
        }

        #[unsafe(no_mangle)]
        extern "C" fn sched_experts(ptr: u32, len: u32) -> u64 {
            // SAFETY: the mechanism's message buffer lives for this call.
            let bytes = unsafe { $crate::abi::message(ptr, len) };
            let load: $crate::ExpertLoad = $crate::abi::experts_message(bytes);
            let order: $crate::ExpertOrder =
                policy(|policy| <$policy as $crate::Policy>::experts(policy, &load));
            $crate::abi::experts_reply(&order)
        }

        #[unsafe(no_mangle)]
        extern "C" fn sched_limits() -> u64 {
            let limits: $crate::Limits = policy(|policy| <$policy as $crate::Policy>::limits(policy));
            $crate::abi::limits_reply(&limits)
        }
    };
}

// The message and reply helpers the macro names. They exist so that the macro
// expands into calls to one place, where a malformed message is a trap and a
// reply is encoded with the wire's encoding.

/// Decodes the message of `sched_init`.
pub fn init_message(bytes: &[u8]) -> Init {
    read(bytes, "init")
}

/// Decodes the message of `sched_observe`.
pub fn events_message(bytes: &[u8]) -> Vec<Event> {
    read(bytes, "observe")
}

/// Decodes the message of `sched_dispatch` and `sched_warm`.
pub fn room_message(bytes: &[u8]) -> Room {
    read(bytes, "dispatch")
}

/// Decodes the message of `sched_reclaim`.
pub fn shortage_message(bytes: &[u8]) -> Shortage {
    read(bytes, "reclaim")
}

/// Decodes the message of `sched_route`, a `(RequestView, [RankView])` tuple
/// as design §4.4's signature gives it.
pub fn route_message(bytes: &[u8]) -> (RequestView, Vec<RankView>) {
    read(bytes, "route")
}

/// Decodes the message of `sched_place`, a `(CallView, [ReplicaView])` tuple.
pub fn place_message(bytes: &[u8]) -> (CallView, Vec<ReplicaView>) {
    read(bytes, "place")
}

/// Decodes the message of `sched_experts`.
pub fn experts_message(bytes: &[u8]) -> ExpertLoad {
    read(bytes, "experts")
}

/// Encodes the reply of `sched_dispatch`.
pub fn dispatch_reply(plan: &Plan) -> u64 {
    reply(plan, "dispatch")
}

/// Encodes the reply of `sched_reclaim`.
pub fn victims_reply(victims: &[Victim]) -> u64 {
    reply(&victims, "reclaim")
}

/// Encodes the reply of `sched_warm`.
pub fn pages_reply(pages: &[crate::PageId]) -> u64 {
    reply(&pages, "warm")
}

/// Encodes the reply of `sched_experts`.
pub fn experts_reply(order: &ExpertOrder) -> u64 {
    reply(order, "experts")
}

/// Encodes the reply of `sched_limits`.
pub fn limits_reply(limits: &Limits) -> u64 {
    reply(limits, "limits")
}

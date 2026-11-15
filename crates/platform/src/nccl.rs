//! Stream-ordered NCCL communicators.

mod ffi;

use std::ffi::c_void;

use ffi::{Handle, api};

use crate::sys::nccl as sys;
use crate::{DeviceId, DevicePtr, DeviceRange, Error, PrimaryContext, Result, Stream};

/// `graphUsageMode` 1: the communicator's collectives are captured into CUDA graphs, at most one
/// such graph is in flight at a time, and its eager collectives are ordered behind that graph by
/// the stream they share.
///
/// NCCL's default, 2, also serves graphs and eager launches in flight on unrelated streams. It
/// pays with one event-record node per captured collective, which serializes the collective
/// against launches NCCL cannot see. The driver holds about 2^18 such nodes in one process; a
/// Qwen3-MoE TP2 process held 261,072 and ran, and TP4 failed instantiating a graph at 286,552
/// with `CUDA_ERROR_OUT_OF_MEMORY` and "Failed to instantiate EVENT_RECORD node". A main process
/// holds a graph per bucket per device with a collective after attention and after the experts of
/// every layer, so the default does not fit a node's worth of devices.
///
/// Mode 1 holds here by construction. Each communicator belongs to one runner, which launches its
/// graphs and its eager forwards on its own stream, and a [`Communicator`]'s collectives started
/// on another stream fork from that stream and join it again. In one graph NCCL still orders a
/// communicator's collectives in capture order. `NCCL_GRAPH_MIXING_SUPPORT`, when it is set,
/// overrides this attribute.
const GRAPH_USAGE_ONE_GRAPH: i32 = 1;

/// The 128-byte rendezvous token every rank of a communicator initializes
/// with. Rank 0 creates it and the caller hands it to the other ranks.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct UniqueId(pub [u8; 128]);

impl std::fmt::Debug for UniqueId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("UniqueId(..)")
    }
}

impl UniqueId {
    pub fn new() -> Result<Self> {
        let api = api()?;
        let mut id = sys::ncclUniqueId::default();
        // SAFETY: NCCL writes the 128-byte id.
        api.check(unsafe { (api.get_unique_id)(&mut id) }, "ncclGetUniqueId")?;
        // `c_char` is signed on some targets; the token's bytes are the same either way.
        Ok(Self(id.internal.map(|byte| byte.to_ne_bytes()[0])))
    }

    /// The token as NCCL's ABI passes it.
    fn raw(&self) -> sys::ncclUniqueId {
        sys::ncclUniqueId {
            internal: self.0.map(|byte| std::ffi::c_char::from_ne_bytes([byte])),
        }
    }
}

/// `ncclRedOp_t` (built-in reductions).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum ReduceOp {
    Sum = sys::ncclSum as i32,
    Prod = sys::ncclProd as i32,
    Max = sys::ncclMax as i32,
    Min = sys::ncclMin as i32,
    Avg = sys::ncclAvg as i32,
}

/// One rank's NCCL communicator.
///
/// Every operation is enqueued on the stream passed to it, which belongs to
/// the communicator's device. Collective calls must be issued in the same
/// order on every rank; group several with [`group`].
///
/// # One communicator per device, driven by that device's thread
///
/// The main process drives every GPU of its node from one thread per device
/// (design 2.4), and each of those threads owns its device's communicator and
/// issues all of its collectives. [`Communicator::init`] blocks until every
/// rank of the communicator has joined, so the ranks' threads must call it
/// concurrently: called one after another in a single thread it never returns.
/// [`group`] covers collective operations, not initialization. A communicator
/// is `Send` but not `Sync`: one thread uses it at a time, which the `&mut
/// self` on its operations enforces.
///
/// # A captured graph outlives no communicator
///
/// A graph that captured a collective must be destroyed before its
/// communicator's [`Communicator::destroy`] runs: `ncclCommDestroy` waits for
/// the communicator's work and does not return while such a graph still
/// references it (measured on GB200). The owner's order is therefore: destroy
/// or rebind a captured graph, drain its stream, then destroy the
/// communicator.
///
/// Release is explicit because it is collective: [`Communicator::destroy`]
/// destroys the communicator once every rank's work retired, and
/// [`Communicator::abort`] tears it down locally after a failure. Dropping a
/// communicator that was neither destroyed nor aborted leaks it deliberately:
/// an implicit destroy could wait forever for peers that never arrive, and an
/// abort could free resources that captured graphs still use.
pub struct Communicator {
    raw: Handle,
    context: PrimaryContext,
    rank: i32,
    size: i32,
}

// SAFETY: a communicator may be used from any one thread at a time; NCCL
// forbids concurrent calls on one communicator. It is therefore Send and not
// Sync: a `&Communicator` cannot cross to another thread, and the collectives
// take `&self` because the ownership rules already keep two threads from
// reaching one communicator at once — which is what lets design 4.3's
// `Forward` hold `group: Option<&Communicator>` and issue collectives through
// it.
unsafe impl Send for Communicator {}

impl Communicator {
    /// Joins communicator `id` as `rank` of `size` on `device`. Every rank
    /// calls this once with the same id; the call blocks until all joined.
    ///
    /// The communicator takes NCCL's default configuration, which `NCCL_*`
    /// environment variables tune, except its graph usage mode
    /// ([`GRAPH_USAGE_ONE_GRAPH`]). Its collectives run as NCCL kernels on
    /// SMs: NCCL's copy-engine (zero-CTA) algorithms apply only to buffers
    /// registered as symmetric windows, and this type registers none.
    pub fn init(device: DeviceId, size: i32, rank: i32, id: &UniqueId) -> Result<Self> {
        if size <= 0 || !(0..size).contains(&rank) {
            return Err(Error::invalid(format!(
                "rank {rank} is outside a communicator of {size} ranks"
            )));
        }
        let api = api()?;
        let context = PrimaryContext::retain(device)?;
        let _guard = context.enter()?;

        let mut raw = std::ptr::null_mut();
        let mut config = sys::ncclConfig_t::initializer();
        config.graphUsageMode = GRAPH_USAGE_ONE_GRAPH;
        // SAFETY: NCCL reads the config, which outlives the call, and writes one handle.
        let status =
            unsafe { (api.comm_init_rank_config)(&mut raw, size, id.raw(), rank, &mut config) };
        if let Err(error) = api.check(status, "ncclCommInitRankConfig") {
            if !raw.is_null() {
                // SAFETY: a partially initialized communicator is owned here.
                let _ = unsafe { (api.comm_abort)(raw) };
            }
            return Err(error);
        }

        Ok(Self {
            raw,
            context,
            rank,
            size,
        })
    }

    pub fn device(&self) -> DeviceId {
        self.context.device()
    }

    pub fn rank(&self) -> i32 {
        self.rank
    }

    pub fn size(&self) -> i32 {
        self.size
    }

    /// Reduces `count` elements from every rank into every rank's `recv` (in
    /// place when both ranges start at the same address).
    ///
    /// `dtype` is the elements' `ncclDataType_t` code. Each range states how
    /// many bytes are valid from its address. The call checks, before any NCCL
    /// call, that `stream` belongs to this communicator's device, that `dtype`
    /// is a code of `nccl.h`, and that `count` elements of it fit both ranges; the caller keeps the storage alive until the stream (or graph
    /// replay) completes.
    ///
    /// Takes `&self` so that a caller may hold the group as a shared reference
    /// (design 4.3's `Forward`); the type is not `Sync`, so a shared reference
    /// never reaches a second thread, which is the one-thread-at-a-time rule
    /// NCCL requires of a communicator.
    pub fn all_reduce(
        &self,
        stream: &Stream,
        send: DeviceRange,
        recv: DeviceRange,
        count: usize,
        dtype: i32,
        op: ReduceOp,
    ) -> Result<()> {
        self.ready(stream, "ncclAllReduce")?;
        let needed = elements_bytes(count, dtype)?;
        require_room(send.bytes(), needed, "send")?;
        require_room(recv.bytes(), needed, "recv")?;
        // NCCL launches on `stream`, which the driver accepts only with the
        // stream's context current on the calling thread; entering it here is
        // what lets any thread issue a collective.
        let _guard = self.context.enter()?;

        let api = api()?;
        // SAFETY: the caller's ranges hold `count` elements of `dtype` and stay
        // valid until the stream completes, which the checks above establish.
        let status = unsafe {
            (api.all_reduce)(
                pointer(send.ptr()),
                pointer(recv.ptr()),
                count,
                dtype,
                op as i32,
                self.raw,
                stream.handle(),
            )
        };
        api.check(status, "ncclAllReduce")
    }

    /// Gathers `count` elements from every rank into `recv`, rank-major.
    ///
    /// The checks are [`Self::all_reduce`]'s, with `count` elements required in
    /// `send` and `size * count` in `recv`.
    pub fn all_gather(
        &self,
        stream: &Stream,
        send: DeviceRange,
        recv: DeviceRange,
        count: usize,
        dtype: i32,
    ) -> Result<()> {
        self.ready(stream, "ncclAllGather")?;
        require_room(send.bytes(), elements_bytes(count, dtype)?, "send")?;
        require_room(recv.bytes(), ranks_bytes(count, dtype, self.size)?, "recv")?;
        // NCCL launches on `stream`, which the driver accepts only with the
        // stream's context current on the calling thread; entering it here is
        // what lets any thread issue a collective.
        let _guard = self.context.enter()?;

        let api = api()?;
        // SAFETY: as in `all_reduce`, with `size * count` elements in `recv`.
        let status = unsafe {
            (api.all_gather)(
                pointer(send.ptr()),
                pointer(recv.ptr()),
                count,
                dtype,
                self.raw,
                stream.handle(),
            )
        };
        api.check(status, "ncclAllGather")
    }

    /// Sends block `j` (`count` elements) of `send` to rank `j` and receives
    /// rank `j`'s block into block `j` of `recv`.
    ///
    /// The checks are [`Self::all_reduce`]'s, with `size * count` elements
    /// required in both ranges.
    pub fn all_to_all(
        &self,
        stream: &Stream,
        send: DeviceRange,
        recv: DeviceRange,
        count: usize,
        dtype: i32,
    ) -> Result<()> {
        self.ready(stream, "ncclAlltoAll")?;
        let needed = ranks_bytes(count, dtype, self.size)?;
        require_room(send.bytes(), needed, "send")?;
        require_room(recv.bytes(), needed, "recv")?;
        // NCCL launches on `stream`, which the driver accepts only with the
        // stream's context current on the calling thread; entering it here is
        // what lets any thread issue a collective.
        let _guard = self.context.enter()?;

        let api = api()?;
        // SAFETY: as in `all_reduce`, with `size * count` elements in both.
        let status = unsafe {
            (api.all_to_all)(
                pointer(send.ptr()),
                pointer(recv.ptr()),
                count,
                dtype,
                self.raw,
                stream.handle(),
            )
        };
        api.check(status, "ncclAlltoAll")
    }

    /// Copies `count` elements of `root`'s `send` into every rank's `recv`.
    ///
    /// The checks are [`Self::all_reduce`]'s, and `root` must be one of this
    /// communicator's ranks; `send` is read on `root` only but must hold
    /// `count` elements on every rank since the same call is posted there.
    pub fn broadcast(
        &self,
        stream: &Stream,
        send: DeviceRange,
        recv: DeviceRange,
        count: usize,
        dtype: i32,
        root: i32,
    ) -> Result<()> {
        self.ready(stream, "ncclBroadcast")?;
        self.require_rank(root, "root")?;
        let needed = elements_bytes(count, dtype)?;
        require_room(send.bytes(), needed, "send")?;
        require_room(recv.bytes(), needed, "recv")?;
        // NCCL launches on `stream`, which the driver accepts only with the
        // stream's context current on the calling thread; entering it here is
        // what lets any thread issue a collective.
        let _guard = self.context.enter()?;

        let api = api()?;
        // SAFETY: as in `all_reduce`, with `count` elements in both ranges.
        let status = unsafe {
            (api.broadcast)(
                pointer(send.ptr()),
                pointer(recv.ptr()),
                count,
                dtype,
                root,
                self.raw,
                stream.handle(),
            )
        };
        api.check(status, "ncclBroadcast")
    }

    /// Sends `count` elements to `peer`, which posts a matching
    /// [`Self::recv`].
    ///
    /// The checks are [`Self::all_reduce`]'s, and `peer` must be one of this
    /// communicator's ranks.
    pub fn send(
        &self,
        stream: &Stream,
        data: DeviceRange,
        count: usize,
        dtype: i32,
        peer: i32,
    ) -> Result<()> {
        self.ready(stream, "ncclSend")?;
        self.require_rank(peer, "peer")?;
        require_room(data.bytes(), elements_bytes(count, dtype)?, "data")?;
        // NCCL launches on `stream`, which the driver accepts only with the
        // stream's context current on the calling thread; entering it here is
        // what lets any thread issue a collective.
        let _guard = self.context.enter()?;

        let api = api()?;
        // SAFETY: as in `all_reduce`, with `count` elements in `data`.
        let status = unsafe {
            (api.send)(
                pointer(data.ptr()),
                count,
                dtype,
                peer,
                self.raw,
                stream.handle(),
            )
        };
        api.check(status, "ncclSend")
    }

    /// Receives `count` elements from `peer`.
    ///
    /// The checks are [`Self::all_reduce`]'s, and `peer` must be one of this
    /// communicator's ranks.
    pub fn recv(
        &self,
        stream: &Stream,
        data: DeviceRange,
        count: usize,
        dtype: i32,
        peer: i32,
    ) -> Result<()> {
        self.ready(stream, "ncclRecv")?;
        self.require_rank(peer, "peer")?;
        require_room(data.bytes(), elements_bytes(count, dtype)?, "data")?;
        // NCCL launches on `stream`, which the driver accepts only with the
        // stream's context current on the calling thread; entering it here is
        // what lets any thread issue a collective.
        let _guard = self.context.enter()?;

        let api = api()?;
        // SAFETY: as in `all_reduce`, with `count` elements in `data`.
        let status = unsafe {
            (api.recv)(
                pointer(data.ptr()),
                count,
                dtype,
                peer,
                self.raw,
                stream.handle(),
            )
        };
        api.check(status, "ncclRecv")
    }

    /// Fails unless `stream` belongs to this communicator's device.
    fn ready(&self, stream: &Stream, call: &'static str) -> Result<()> {
        if stream.device() != self.device() {
            return Err(Error::invalid(format!(
                "{call} on the communicator of {} takes a stream of that device, not of {}",
                self.device(),
                stream.device()
            )));
        }
        Ok(())
    }

    /// Fails unless `rank` is one of this communicator's ranks.
    fn require_rank(&self, rank: i32, what: &'static str) -> Result<()> {
        if !(0..self.size).contains(&rank) {
            return Err(Error::invalid(format!(
                "{what} {rank} is outside a communicator of {} ranks",
                self.size
            )));
        }
        Ok(())
    }

    /// Splits a subgroup out of this communicator (`ncclCommSplit`), which is
    /// how a world communicator becomes the TP, DP and expert groups the
    /// deployment's blocks run on.
    ///
    /// `ranks` are the parent ranks of the subgroup this rank belongs to, in
    /// the order their new ranks take; `rank` is this communicator's rank in
    /// the parent. The colour is derived from the subgroup itself — its
    /// smallest parent rank — so one call serves every subgroup of a world:
    /// every member passes the same list, and disjoint subgroups get distinct
    /// colours. A rank that is not in `ranks` still calls this, because the
    /// split is collective over the parent; it passes `NCCL_SPLIT_NOCOLOR` and
    /// answers `None`, since NCCL hands it no communicator.
    ///
    /// The child is destroyed before its parent: a parent cannot retire while
    /// a child still communicates. `ranks` must be non-empty and must contain
    /// no duplicate.
    pub fn split(&self, ranks: &[i32], rank: i32) -> Result<Option<Self>> {
        let position = ranks.iter().position(|parent| *parent == rank);
        let (color, key) = match position {
            Some(position) => {
                let smallest = ranks
                    .iter()
                    .copied()
                    .min()
                    .ok_or_else(|| Error::invalid("a split needs the ranks of its subgroup"))?;
                (smallest, position as i32)
            }
            None => (sys::NCCL_SPLIT_NOCOLOR, 0),
        };
        let api = api()?;
        let _guard = self.context.enter()?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: the parent is live; NCCL writes the child handle (NULL when
        // this rank does not join) and reads no config because none is given.
        let status =
            unsafe { (api.comm_split)(self.raw, color, key, &mut raw, std::ptr::null_mut()) };
        api.check(status, "ncclCommSplit")?;
        if raw.is_null() {
            return Ok(None);
        }
        Ok(Some(Self {
            raw,
            context: self.context.clone(),
            rank: position.ok_or_else(|| {
                Error::invalid("NCCL returned a communicator to a rank that did not join")
            })? as i32,
            size: ranks.len() as i32,
        }))
    }

    /// Destroys the communicator. Collective:
    /// every rank calls it after all work using the communicator, including
    /// graphs that captured it, has retired. On failure the communicator's
    /// remaining resources are leaked.
    pub fn destroy(self) -> Result<()> {
        let api = api()?;
        let _guard = self.context.enter()?;

        // SAFETY: the communicator is owned and destroyed once.
        api.check(unsafe { (api.comm_destroy)(self.raw) }, "ncclCommDestroy")
    }
}

/// The width in bytes of one element of the `ncclDataType_t` code `dtype`, as `nccl.h` defines
/// the code.
///
/// The code is the caller's, derived from its own element type; this is the one place the crate
/// reads it, to check a range's room before NCCL addresses it. A code `nccl.h` does not define is
/// refused rather than passed on.
fn width(dtype: i32) -> Result<usize> {
    let code = sys::ncclDataType_t::try_from(dtype)
        .map_err(|_| Error::invalid(format!("{dtype} is no ncclDataType_t code of nccl.h")))?;
    match code {
        sys::ncclInt8 | sys::ncclUint8 | sys::ncclFloat8e4m3 | sys::ncclFloat8e5m2 => Ok(1),
        sys::ncclFloat16 | sys::ncclBfloat16 => Ok(2),
        sys::ncclInt32 | sys::ncclUint32 | sys::ncclFloat32 => Ok(4),
        sys::ncclInt64 | sys::ncclUint64 | sys::ncclFloat64 => Ok(8),
        _ => Err(Error::invalid(format!(
            "{dtype} is no ncclDataType_t code of nccl.h"
        ))),
    }
}

/// The bytes `count` elements of the NCCL type `dtype` occupy, without overflowing.
fn elements_bytes(count: usize, dtype: i32) -> Result<usize> {
    count.checked_mul(width(dtype)?).ok_or_else(|| {
        Error::invalid(format!(
            "{count} elements of NCCL type {dtype} overflow the byte count"
        ))
    })
}

/// The bytes `size * count` elements of the NCCL type `dtype` occupy, without overflowing.
fn ranks_bytes(count: usize, dtype: i32, size: i32) -> Result<usize> {
    elements_bytes(count, dtype)?
        .checked_mul(size as usize)
        .ok_or_else(|| {
            Error::invalid(format!(
                "{count} elements of NCCL type {dtype} on {size} ranks overflow the byte count"
            ))
        })
}

/// Fails unless `bytes` states at least `needed` bytes from `what`'s pointer.
fn require_room(bytes: usize, needed: usize, what: &'static str) -> Result<()> {
    if bytes < needed {
        return Err(Error::invalid(format!(
            "the {what} range states {bytes} bytes, but the collective needs {needed}"
        )));
    }
    Ok(())
}

fn pointer(address: DevicePtr) -> Handle {
    address.address() as usize as *mut c_void
}

/// Runs `submit` inside `ncclGroupStart` / `ncclGroupEnd`, so that its
/// operations (for example matching sends and receives) launch together.
/// The group is ended even if `submit` fails; its error is reported first.
pub fn group<R>(submit: impl FnOnce() -> Result<R>) -> Result<R> {
    let api = api()?;
    // SAFETY: group calls take no arguments and nest per thread.
    api.check(unsafe { (api.group_start)() }, "ncclGroupStart")?;
    let result = submit();
    // SAFETY: balances the start above on the same thread.
    let ended = api.check(unsafe { (api.group_end)() }, "ncclGroupEnd");
    let value = result?;
    ended?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::{Communicator, sys};

    /// The main process moves a communicator to the thread that drives its
    /// device; it never shares one between threads at once.
    #[test]
    fn a_communicator_moves_to_its_device_thread() {
        fn movable<T: Send>() {}
        movable::<Communicator>();
    }

    /// A range's room is checked at the width `nccl.h` gives each type code,
    /// and a code it does not define is refused before NCCL sees it.
    #[test]
    fn a_type_code_has_the_width_nccl_gives_it() {
        for (code, width) in [
            (sys::ncclInt8, 1),
            (sys::ncclFloat16, 2),
            (sys::ncclBfloat16, 2),
            (sys::ncclFloat32, 4),
            (sys::ncclInt64, 8),
            (sys::ncclFloat8e4m3, 1),
        ] {
            assert_eq!(super::width(code as i32).unwrap(), width, "code {code}");
        }
        assert!(super::width(-1).is_err());
        assert!(super::width(sys::ncclNumTypes as i32).is_err());
        assert_eq!(
            super::ranks_bytes(3, sys::ncclFloat32 as i32, 4).unwrap(),
            48
        );
    }

    /// The rendezvous token every rank exchanges is the header's
    /// `ncclUniqueId`, byte for byte: another size would be another token
    /// than the other ranks hold.
    #[test]
    fn the_unique_id_is_the_headers_token() {
        assert_eq!(
            std::mem::size_of::<super::UniqueId>(),
            sys::NCCL_UNIQUE_ID_BYTES as usize
        );
        assert_eq!(
            std::mem::size_of::<super::UniqueId>(),
            std::mem::size_of::<sys::ncclUniqueId>()
        );
    }
}

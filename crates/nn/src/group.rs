//! The communicator a partitioned block runs its collectives through.
//!
//! The design's [`Forward`](crate::Forward) carries `Option<&Communicator>` (specs/worker.md 4.3):
//! a non-owning borrowing of the NCCL communicator the executor created for this rank. The CUDA
//! boundary's collectives take `&mut Communicator` because NCCL calls into one communicator must
//! not interleave, while a forward holds its resources immutably. [`Group`] is the box the executor
//! puts its communicator in: it owns the communicator, serializes access to it on the one executor
//! thread, and offers the collectives a layer needs through `&self`.
//!
//! One executor thread issues every job of its device (specs/worker.md 2.4), so the lock here is not
//! contention: it is the type-level statement that a forward borrows the group and does not own it,
//! and it is what lets the runners of one device share a group by `Arc` across the move to their
//! device thread.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

use ops::platform::{CaptureStatus, DeviceRange, Event, EventFlags, PlatformDevicePtr};
use ops::{CollectiveType, Communicator, ReduceOp, Stream};

use crate::error::{Error, Result};
use crate::tensor::TensorView;

/// The bytes `view` addresses, as the collective's range type wants them.
fn range(view: TensorView, bytes: usize) -> DeviceRange {
    DeviceRange::new(PlatformDevicePtr::from_address(view.ptr.address()), bytes)
}

/// The priority of a group's lane: the driver clamps it to the device's highest.
const LANE_PRIORITY: i32 = -5;

/// How many collectives a group holds in flight on its lane.
const LANDINGS: usize = 16;

/// A collective running on a group's lane, until a stream takes it up ([`Group::finish`]).
#[must_use = "what a transfer writes is readable only on a stream that finished it"]
pub struct Transfer {
    landing: usize,
}

/// A rank's NCCL communicator, borrowed by forwards.
pub struct Group {
    communicator: Mutex<Communicator>,
    ranks: u64,
    /// The stream a started collective runs on, beside the forward's own, so a layer's compute
    /// overlaps the transfer. Inside a capture the events below make it a branch of the graph.
    lane: Stream,
    /// Orders the lane after what the forward issued so far.
    fork: Event,
    /// Each started collective's end, which a forward's stream waits for.
    landings: Vec<Event>,
    /// The landing the next collective takes.
    next: AtomicUsize,
}

impl Group {
    /// The group of `communicator`, whose member count is `ranks`.
    pub fn new(communicator: Communicator, ranks: u64) -> Result<Self> {
        // The lane's kernels take an SM as soon as one frees: they are small and everything on the
        // forward's stream waits for them sooner or later, and a transfer queued behind a whole
        // projection would run after it instead of beside it.
        let lane =
            Stream::with_priority(communicator.device(), LANE_PRIORITY).map_err(Error::Platform)?;
        Ok(Self {
            communicator: Mutex::new(communicator),
            ranks,
            lane,
            fork: Event::new(EventFlags::default()),
            landings: (0..LANDINGS)
                .map(|_| Event::new(EventFlags::default()))
                .collect(),
            next: AtomicUsize::new(0),
        })
    }

    /// How many ranks the group has.
    pub const fn ranks(&self) -> u64 {
        self.ranks
    }

    /// This rank's index, as NCCL reports it.
    pub fn rank(&self) -> i32 {
        self.locked().rank()
    }

    /// Sum every rank's `view` into every rank's `view`, in place.
    ///
    /// `view` is the same shape and dtype on every rank; the collective is issued on `stream`.
    pub fn all_reduce_sum(&self, view: TensorView, stream: &Stream) -> Result<()> {
        self.all_reduce(view, view, ReduceOp::Sum, stream)
    }

    /// Reduce `send` from every rank into every rank's `recv` with `op`.
    pub fn all_reduce(
        &self,
        send: TensorView,
        recv: TensorView,
        op: ReduceOp,
        stream: &Stream,
    ) -> Result<()> {
        let (count, dtype) = self.collective(send, recv, "all_reduce")?;
        let bytes = dtype.bytes() * count;
        self.locked()
            .all_reduce(
                stream,
                range(send, bytes),
                range(recv, bytes),
                count,
                dtype,
                op,
            )
            .map_err(Error::Platform)?;
        Self::settled(stream)
    }

    /// Gather `send` from every rank into `recv`, rank-major.
    ///
    /// `recv` holds `ranks` times `send`'s elements, in rank order.
    pub fn all_gather(&self, send: TensorView, recv: TensorView, stream: &Stream) -> Result<()> {
        let (count, dtype) = self.collective(send, send, "all_gather")?;
        let expected = count
            .checked_mul(usize::try_from(self.ranks).unwrap_or(usize::MAX))
            .ok_or_else(|| Error::shape("Group::all_gather", "the gathered count overflows"))?;
        if recv.numel() != expected as u64 || recv.dtype != send.dtype {
            return Err(Error::shape(
                "Group::all_gather",
                format!(
                    "receives {} {} elements, {expected} of {} expected",
                    recv.numel(),
                    recv.dtype,
                    send.dtype
                ),
            ));
        }
        let bytes = dtype.bytes();
        self.locked()
            .all_gather(
                stream,
                range(send, bytes * count),
                range(recv, bytes * expected),
                count,
                dtype,
            )
            .map_err(Error::Platform)?;
        Self::settled(stream)
    }

    /// Exchange equal blocks with every rank: block `j` of `send` goes to rank `j`, and block `j` of
    /// `recv` is what rank `j` sent here.
    ///
    /// `send` and `recv` are the whole buffers, `ranks` blocks each, and NCCL's `count` is one
    /// block's elements (`notes/exec/nccl-alltoall-count.md`).
    pub fn all_to_all(&self, send: TensorView, recv: TensorView, stream: &Stream) -> Result<()> {
        let (elements, dtype) = self.collective(send, recv, "all_to_all")?;
        let ranks = usize::try_from(self.ranks).unwrap_or(usize::MAX);
        if recv.numel() != send.numel() || !elements.is_multiple_of(ranks.max(1)) {
            return Err(Error::shape(
                "Group::all_to_all",
                format!(
                    "sends {elements} elements and receives {}, which are not {ranks} equal blocks",
                    recv.numel()
                ),
            ));
        }
        let bytes = dtype.bytes() * elements;
        self.locked()
            .all_to_all(
                stream,
                range(send, bytes),
                range(recv, bytes),
                elements / ranks.max(1),
                dtype,
            )
            .map_err(Error::Platform)?;
        Self::settled(stream)
    }

    /// Starts [`Self::all_to_all`] on the group's lane, after everything `stream` has issued.
    ///
    /// `stream` goes on at once, and what `recv` holds is readable on a stream that took the
    /// transfer up ([`Self::finish`]). The buffers belong to the transfer until then.
    pub fn start_all_to_all(
        &self,
        send: TensorView,
        recv: TensorView,
        stream: &Stream,
    ) -> Result<Transfer> {
        self.fork.record(stream).map_err(Error::Platform)?;
        self.lane
            .wait_event(&self.fork)
            .map_err(Error::Platform)?;
        self.all_to_all(send, recv, &self.lane)?;
        let landing = self.next.fetch_add(1, Ordering::Relaxed) % LANDINGS;
        self.landings[landing]
            .record(&self.lane)
            .map_err(Error::Platform)?;
        Ok(Transfer { landing })
    }

    /// Starts delivering rows `first..first + count` of this rank's rows to every member, and
    /// theirs here, in place in `gathered`.
    ///
    /// `gathered` is the whole sequence `[members * owner_rows, width]`, member by member, and this
    /// rank's rows of the interval are already in its own block. A row lands where the sequence
    /// holds it, so an interval of every member arrives without a copy and the sequence is whole
    /// once each interval's transfer is finished.
    pub fn start_gather_rows(
        &self,
        gathered: TensorView,
        owner_rows: u64,
        first: u64,
        count: u64,
        stream: &Stream,
    ) -> Result<Transfer> {
        let width = gathered.shape.last();
        let (elements, dtype) = self.collective(gathered, gathered, "gather_rows")?;
        let elements = elements.min(usize::try_from(count * width).unwrap_or(usize::MAX));
        let bytes = dtype.bytes() * elements;
        let block = |member: u64| {
            let at = gathered.offset_elements((member * owner_rows + first) * width);
            range(gathered.at(at), bytes)
        };
        self.fork.record(stream).map_err(Error::Platform)?;
        self.lane
            .wait_event(&self.fork)
            .map_err(Error::Platform)?;
        {
            let communicator = self.locked();
            let rank = communicator.rank();
            ops::platform::collective_group(|| {
                for peer in 0..self.ranks as i32 {
                    if peer != rank {
                        communicator.send(&self.lane, block(rank as u64), elements, dtype, peer)?;
                        communicator.recv(&self.lane, block(peer as u64), elements, dtype, peer)?;
                    }
                }
                Ok(())
            })
            .map_err(Error::Platform)?;
        }
        Self::settled(&self.lane)?;
        let landing = self.next.fetch_add(1, Ordering::Relaxed) % LANDINGS;
        self.landings[landing]
            .record(&self.lane)
            .map_err(Error::Platform)?;
        Ok(Transfer { landing })
    }

    /// Orders `stream` after `transfer`.
    pub fn finish(&self, transfer: Transfer, stream: &Stream) -> Result<()> {
        stream
            .wait_event(&self.landings[transfer.landing])
            .map_err(Error::Platform)
    }

    /// Run `call` with the underlying communicator, for an operator whose collective is not one of
    /// the three above (an expert transport).
    pub fn with<R>(&self, call: impl FnOnce(&mut Communicator) -> Result<R>) -> Result<R> {
        call(&mut self.locked())
    }

    /// Waits for a collective issued outside a capture, so the walk goes on with none in flight.
    ///
    /// NCCL's kernel spins on its device until every member reaches the collective. The first
    /// `cuKernelSetAttribute` of a kernel, which cuBLASLt issues inside a first `cublasLtMatmul` and
    /// a native launch inside `cudaFuncSetAttribute`, takes a lock of the whole process and then
    /// waits for its device to drain. Every rank of a host is a thread of one process, so a rank
    /// that walked on past a collective and reached such a call holds that lock for a kernel that
    /// needs its peers, while the peers queue on the lock in front of the same collective. The
    /// warm walk of a bucket is where those first calls happen, and the walks of a group's ranks
    /// are not in step, because a rank's rows of each modality are its own.
    ///
    /// A collective inside a capture runs in a replay, by which time every kernel is ready, so it
    /// is not waited for.
    fn settled(stream: &Stream) -> Result<()> {
        if stream.capture_status().map_err(Error::Platform)? == CaptureStatus::None {
            stream.synchronize().map_err(Error::Platform)?;
        }
        Ok(())
    }

    /// The communicator, for one call. A panic while it was held leaves nothing half-issued that
    /// this type could repair, so a poisoned lock answers the communicator as it stands.
    fn locked(&self) -> MutexGuard<'_, Communicator> {
        self.communicator
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The element count of one rank and the collective's element type.
    fn collective(
        &self,
        send: TensorView,
        recv: TensorView,
        collective: &'static str,
    ) -> Result<(usize, CollectiveType)> {
        if send.dtype != recv.dtype {
            return Err(Error::shape(
                collective,
                format!("collects {} against {}", send.dtype, recv.dtype),
            ));
        }
        let count = usize::try_from(send.numel())
            .map_err(|_| Error::shape(collective, "a collective element count above usize::MAX"))?;
        let dtype = ops::collective_type(send.dtype).ok_or_else(|| {
            Error::Unsupported(format!(
                "{collective}: {} has no NCCL element type",
                send.dtype
            ))
        })?;
        Ok((count, dtype))
    }
}

/// The NCCL element type of an operator ABI element type.
///
/// The mapping is `ops`': an element type is named once, in the crate that owns the ABI, so a
/// collective and a kernel program cannot disagree about what `bf16` means.
pub use ops::collective_type;

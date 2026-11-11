//! The communicator a partitioned block runs its collectives through.
//!
//! The design's [`Forward`](crate::Forward) carries `Option<&Communicator>`:
//! a non-owning borrowing of the NCCL communicator the worker created for this rank. The CUDA
//! boundary's collectives take `&mut Communicator` because NCCL calls into one communicator must
//! not interleave, while a forward holds its resources immutably. [`Group`] is the box the worker
//! puts its communicator in: it owns the communicator, serializes access to it on the one worker
//! thread, and offers the collectives a forward issues through `&self`.
//!
//! A layer never calls a collective here: it calls the collective of its [`Forward`], which issues
//! it through this group on the forward's stream, or, on a recording forward, checks its views
//! against the forward's split and issues nothing.
//!
//! [`Forward`]: crate::Forward
//!
//! One worker thread issues every job of its device, so the lock here is not
//! contention: it is the type-level statement that a forward borrows the group and does not own it,
//! and it is what lets the runners of one device share a group by `Arc` across the move to their
//! device thread.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

use ops::platform::{CaptureStatus, DeviceRange, Event, EventFlags, PlatformDevicePtr};
use ops::{CollectiveType, Communicator, ReduceOp, Stream};

use crate::error::{Error, Result};
use crate::tensor::Operand;

/// The bytes `view` addresses, as the collective's range type wants them.
fn range(view: Operand, bytes: usize) -> DeviceRange {
    DeviceRange::new(PlatformDevicePtr::from_address(view.ptr.address()), bytes)
}

/// The priority of a group's lane: the driver clamps it to the device's highest.
const LANE_PRIORITY: i32 = -5;

/// How many collectives a group holds in flight on its lane.
const LANDINGS: usize = 16;

/// A collective running on a group's lane, until a stream takes it up
/// ([`Forward::finish`](crate::Forward::finish)).
///
/// A recording forward answers a transfer that holds no landing: it issued nothing, so finishing it
/// orders nothing.
#[must_use = "what a transfer writes is readable only on a stream that finished it"]
pub struct Transfer {
    landing: Option<usize>,
}

impl Transfer {
    /// The transfer a recording forward answers for a collective it did not issue.
    pub(crate) const fn recorded() -> Self {
        Self { landing: None }
    }
}

/// The element count of one rank and the collective's element type, refusing views the
/// communicator could not move.
pub(crate) fn collective(
    send: Operand,
    recv: Operand,
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

/// The element counts of an all-gather over `ranks` members, `send`'s and the `ranks` times as
/// many `recv` must hold, and its element type.
pub(crate) fn gathered(
    send: Operand,
    recv: Operand,
    ranks: u64,
) -> Result<(usize, usize, CollectiveType)> {
    let (count, dtype) = collective(send, send, "all_gather")?;
    let expected = count
        .checked_mul(usize::try_from(ranks).unwrap_or(usize::MAX))
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
    Ok((count, expected, dtype))
}

/// The element count of an all-to-all over `ranks` members, which both buffers hold as `ranks`
/// equal blocks, and its element type.
pub(crate) fn exchanged(
    send: Operand,
    recv: Operand,
    ranks: u64,
) -> Result<(usize, CollectiveType)> {
    let (elements, dtype) = collective(send, recv, "all_to_all")?;
    let ranks = usize::try_from(ranks).unwrap_or(usize::MAX);
    if recv.numel() != send.numel() || !elements.is_multiple_of(ranks.max(1)) {
        return Err(Error::shape(
            "Group::all_to_all",
            format!(
                "sends {elements} elements and receives {}, which are not {ranks} equal blocks",
                recv.numel()
            ),
        ));
    }
    Ok((elements, dtype))
}

/// A rank's place in a partitioned block's group, and the NCCL communicator its collectives are
/// issued through, borrowed by forwards.
///
/// A group of a recorded start ([`Group::recording`]) states the split alone: a recording forward
/// reads its rank and member count and issues nothing, so no communicator is formed for it, and
/// every collective called on it is refused.
pub struct Group {
    /// This rank's index in the group.
    rank: i32,
    ranks: u64,
    /// What issues the group's collectives; absent on a recording group.
    issue: Option<Issue>,
}

/// The communicator of a group that issues, with the lane its started collectives run on.
struct Issue {
    communicator: Mutex<Communicator>,
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
            rank: communicator.rank(),
            ranks,
            issue: Some(Issue {
                communicator: Mutex::new(communicator),
                lane,
                fork: Event::new(EventFlags::default()),
                landings: (0..LANDINGS)
                    .map(|_| Event::new(EventFlags::default()))
                    .collect(),
                next: AtomicUsize::new(0),
            }),
        })
    }

    /// The split of member `rank` of a group of `ranks`, with no communicator: what the runners of
    /// a recorded start walk with, whose forwards record and issue no collective.
    ///
    /// # Errors
    ///
    /// When `rank` is not below `ranks`, or is past what NCCL addresses.
    pub fn recording(rank: u64, ranks: u64) -> Result<Self> {
        if rank >= ranks {
            return Err(Error::shape(
                "Group::recording",
                format!("rank {rank} is not a member of a group of {ranks}"),
            ));
        }
        let rank = i32::try_from(rank)
            .map_err(|_| Error::shape("Group::recording", "a rank past what NCCL addresses"))?;
        Ok(Self {
            rank,
            ranks,
            issue: None,
        })
    }

    /// How many ranks the group has.
    pub const fn ranks(&self) -> u64 {
        self.ranks
    }

    /// This rank's index, as NCCL reports it.
    pub fn rank(&self) -> i32 {
        self.rank
    }

    /// Sum every rank's `view` into every rank's `view`, in place.
    ///
    /// `view` is the same shape and dtype on every rank; the collective is issued on `stream`.
    pub(crate) fn all_reduce_sum(&self, view: Operand, stream: &Stream) -> Result<()> {
        self.all_reduce(view, view, ReduceOp::Sum, stream)
    }

    /// Reduce `send` from every rank into every rank's `recv` with `op`.
    pub(crate) fn all_reduce(
        &self,
        send: Operand,
        recv: Operand,
        op: ReduceOp,
        stream: &Stream,
    ) -> Result<()> {
        let (count, dtype) = collective(send, recv, "all_reduce")?;
        let bytes = dtype.bytes() * count;
        self.issue()?
            .locked()
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
    pub(crate) fn all_gather(
        &self,
        send: Operand,
        recv: Operand,
        stream: &Stream,
    ) -> Result<()> {
        let (count, expected, dtype) = gathered(send, recv, self.ranks)?;
        let bytes = dtype.bytes();
        self.issue()?
            .locked()
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
    /// block's elements.
    pub(crate) fn all_to_all(
        &self,
        send: Operand,
        recv: Operand,
        stream: &Stream,
    ) -> Result<()> {
        let (elements, dtype) = exchanged(send, recv, self.ranks)?;
        let ranks = usize::try_from(self.ranks).unwrap_or(usize::MAX);
        let bytes = dtype.bytes() * elements;
        self.issue()?
            .locked()
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
    pub(crate) fn start_all_to_all(
        &self,
        send: Operand,
        recv: Operand,
        stream: &Stream,
    ) -> Result<Transfer> {
        let issue = self.issue()?;
        issue.fork.record(stream).map_err(Error::Platform)?;
        issue
            .lane
            .wait_event(&issue.fork)
            .map_err(Error::Platform)?;
        self.all_to_all(send, recv, &issue.lane)?;
        let landing = issue.next.fetch_add(1, Ordering::Relaxed) % LANDINGS;
        issue.landings[landing]
            .record(&issue.lane)
            .map_err(Error::Platform)?;
        Ok(Transfer {
            landing: Some(landing),
        })
    }

    /// Starts delivering rows `first..first + count` of this rank's rows to every member, and
    /// theirs here, in place in `gathered`.
    ///
    /// `gathered` is the whole sequence `[members * owner_rows, width]`, member by member, and this
    /// rank's rows of the interval are already in its own block. A row lands where the sequence
    /// holds it, so an interval of every member arrives without a copy and the sequence is whole
    /// once each interval's transfer is finished.
    pub(crate) fn start_gather_rows(
        &self,
        gathered: Operand,
        owner_rows: u64,
        first: u64,
        count: u64,
        stream: &Stream,
    ) -> Result<Transfer> {
        let width = gathered.shape.last();
        let (elements, dtype) = collective(gathered, gathered, "gather_rows")?;
        let elements = elements.min(usize::try_from(count * width).unwrap_or(usize::MAX));
        let bytes = dtype.bytes() * elements;
        let block = |member: u64| {
            let at = gathered.offset_elements((member * owner_rows + first) * width);
            range(gathered.at(at), bytes)
        };
        let issue = self.issue()?;
        issue.fork.record(stream).map_err(Error::Platform)?;
        issue
            .lane
            .wait_event(&issue.fork)
            .map_err(Error::Platform)?;
        {
            let communicator = issue.locked();
            let rank = communicator.rank();
            ops::platform::collective_group(|| {
                for peer in 0..self.ranks as i32 {
                    if peer != rank {
                        communicator.send(
                            &issue.lane,
                            block(rank as u64),
                            elements,
                            dtype,
                            peer,
                        )?;
                        communicator.recv(
                            &issue.lane,
                            block(peer as u64),
                            elements,
                            dtype,
                            peer,
                        )?;
                    }
                }
                Ok(())
            })
            .map_err(Error::Platform)?;
        }
        Self::settled(&issue.lane)?;
        let landing = issue.next.fetch_add(1, Ordering::Relaxed) % LANDINGS;
        issue.landings[landing]
            .record(&issue.lane)
            .map_err(Error::Platform)?;
        Ok(Transfer {
            landing: Some(landing),
        })
    }

    /// Orders `stream` after `transfer`; a recorded transfer orders nothing.
    pub(crate) fn finish(&self, transfer: Transfer, stream: &Stream) -> Result<()> {
        let Some(landing) = transfer.landing else {
            return Ok(());
        };
        stream
            .wait_event(&self.issue()?.landings[landing])
            .map_err(Error::Platform)
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

    /// What issues this group's collectives, refused on a recording group: a collective reaches a
    /// group only from a launching forward, and a recorded start launches nothing.
    fn issue(&self) -> Result<&Issue> {
        self.issue.as_ref().ok_or_else(|| {
            Error::Unsupported(
                "a recording group has no communicator and issues no collective".to_owned(),
            )
        })
    }
}

impl Issue {
    /// The communicator, for one call. A panic while it was held leaves nothing half-issued that
    /// this type could repair, so a poisoned lock answers the communicator as it stands.
    fn locked(&self) -> MutexGuard<'_, Communicator> {
        self.communicator
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// The NCCL element type of an operator ABI element type.
///
/// The mapping is `ops`': an element type is named once, in the crate that owns the ABI, so a
/// collective and a kernel program cannot disagree about what `bf16` means.
pub use ops::collective_type;

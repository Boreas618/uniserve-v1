//! Collectives over the context's group.
//!
//! A partitioned forward computes one rank's share and meets the other ranks through
//! collectives: a sum over every rank's buffer, a gather of every rank's buffer in rank order,
//! and an exchange of equal blocks between every pair of ranks. A module reads its place in the
//! group with [`split`] and calls the collective; which communicator and stream carry it is the
//! context's.
//!
//! Under a context that launches, a collective is issued through the context's [`Group`] on the
//! context's stream, and while that stream is being captured it becomes nodes of the graph.
//! Under a meta context it checks its buffers against the split and issues nothing, so a profile
//! run and a `prepare_inputs` forward follow the same shapes with no communicator.
//!
//! A collective may also be **started** beside the stream and **finished** later
//! ([`start_all_to_all`], [`start_gather_rows`], [`finish`]): it runs on the group's side stream
//! while the forward's stream goes on computing, and what it received is readable once the
//! forward's stream finished it.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use ops::platform::{
    CaptureStatus, CollectiveType, Communicator, DeviceRange, Event, EventFlags, PlatformDevicePtr,
    ReduceOp, Stream, collective_group,
};
use ops::{Operand, collective_type};

use crate::context::{self, Context};
use crate::{Error, Result};

/// The priority of a group's side stream: the driver clamps it to the device's highest.
///
/// A started collective's kernels are small and everything on the forward's stream waits for
/// them sooner or later, so they take an SM as soon as one frees instead of queueing behind a
/// whole projection.
const SIDE_PRIORITY: i32 = -5;

/// How many started collectives a group keeps an end event for.
///
/// The events are taken in turn. A collective started after sixteen others that are not yet
/// finished takes the first one's event again, and finishing the first then waits for the later
/// one, which is later on the same side stream: the order still holds.
const ENDS: usize = 16;

/// A rank's NCCL communicator, with the side stream its started collectives run on.
///
/// Execution creates one group per communicator and sets it on the contexts of the runners that
/// share it ([`Context::set_group`]). The communicator may be called from one thread at a time,
/// which the one executor thread of a device already keeps; the lock states it in the type, so
/// that contexts sharing a group move to their device's thread with their runners.
pub struct Group {
    communicator: Mutex<Communicator>,
    /// The stream a started collective runs on, beside the forward's own, so a module's compute
    /// overlaps the transfer. Inside a capture the events below make it a branch of the graph.
    side: Stream,
    /// Orders the side stream after what the forward's stream issued so far.
    fork: Event,
    /// The end of each started collective, which a forward's stream waits for.
    ends: Vec<Event>,
    /// The end event the next started collective takes, modulo [`ENDS`].
    next: AtomicUsize,
}

impl Group {
    /// The group of `communicator`.
    ///
    /// # Errors
    ///
    /// [`Error::Platform`] when the side stream cannot be created on the communicator's device.
    pub fn new(communicator: Communicator) -> Result<Self> {
        let side = Stream::with_priority(communicator.device(), SIDE_PRIORITY)?;

        Ok(Self {
            communicator: Mutex::new(communicator),
            side,
            fork: Event::new(EventFlags::default()),
            ends: (0..ENDS)
                .map(|_| Event::new(EventFlags::default()))
                .collect(),
            next: AtomicUsize::new(0),
        })
    }

    /// This rank's index in the communicator and the communicator's member count.
    pub(crate) fn place(&self) -> (u64, u64) {
        let communicator = self.communicator();

        // NCCL reports both as non-negative `int`s.
        (
            u64::try_from(communicator.rank()).unwrap_or(0),
            u64::try_from(communicator.size()).unwrap_or(0),
        )
    }

    /// The communicator, for one call. A panic while it was held leaves nothing half-issued that
    /// this type could repair, so a poisoned lock gives the communicator as it stands.
    fn communicator(&self) -> MutexGuard<'_, Communicator> {
        self.communicator
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Runs `issue` on the side stream, ordered after everything `stream` issued so far, and
    /// returns the index of the event that marks its end.
    fn start(&self, stream: &Stream, issue: impl FnOnce(&Stream) -> Result<()>) -> Result<usize> {
        self.fork.record(stream)?;
        self.side.wait_event(&self.fork)?;

        issue(&self.side)?;

        let end = self.next.fetch_add(1, Ordering::Relaxed) % ENDS;
        self.ends[end].record(&self.side)?;
        Ok(end)
    }

    /// Issues the exchange of `block` elements with every rank on `stream`; `send` and `recv`
    /// hold `bytes` bytes each.
    fn all_to_all(
        &self,
        send: Operand,
        recv: Operand,
        (block, bytes, dtype): (usize, usize, CollectiveType),
        stream: &Stream,
    ) -> Result<()> {
        self.communicator().all_to_all(
            stream,
            range(send, bytes),
            range(recv, bytes),
            block,
            dtype,
        )?;
        wait_unless_captured(stream)
    }

    /// Issues, on `stream`, the send of this rank's `elements` elements at `block(rank)` to every
    /// other rank and the receive of theirs at `block(peer)`, as one NCCL group.
    fn exchange_rows(
        &self,
        block: impl Fn(u64) -> DeviceRange,
        (elements, dtype): (usize, CollectiveType),
        ranks: u64,
        stream: &Stream,
    ) -> Result<()> {
        {
            let communicator = self.communicator();
            let rank = communicator.rank();

            collective_group(|| {
                for peer in 0..ranks as i32 {
                    if peer != rank {
                        communicator.send(stream, block(rank as u64), elements, dtype, peer)?;
                        communicator.recv(stream, block(peer as u64), elements, dtype, peer)?;
                    }
                }
                Ok(())
            })?;
        }
        wait_unless_captured(stream)
    }
}

// tensors: the public collectives take their buffers as kernel operands, the form a tensor's
// storage and layout give. The signatures from here to `finish` are the one place a collective's
// argument type is stated to a caller; the checks and the group below work on operands, reading
// an address, an element type and an element count.

/// This rank's index in the group and the group's member count, or `None` when the forward is
/// not partitioned or no context is installed.
///
/// A module's shape arithmetic over a partition (a vocabulary shard, a gather buffer, a rank's
/// share of tiles) reads this, so a meta forward follows the shapes the launching forward of
/// that rank does. A group of one member is a split: its collectives are issued as for any
/// group.
pub fn split() -> Option<(u64, u64)> {
    context::installed()?.state().split
}

/// Sums every rank's `x` into every rank's `x`, in place.
///
/// `x` has the same shape and element type on every rank.
///
/// # Errors
///
/// [`Error::NoContext`]; [`Error::Collective`] when the context carries no group or NCCL has no
/// element type for `x`; [`Error::Platform`] when the collective cannot be issued.
pub fn all_reduce(x: Operand) -> Result<()> {
    let (context, _) = member("all_reduce")?;
    let (count, dtype) = elements(x, x, "all_reduce")?;

    let Some(stream) = context.stream() else {
        return Ok(());
    };
    let group = group(&context, "all_reduce")?;
    let bytes = dtype.bytes() * count;

    group.communicator().all_reduce(
        stream,
        range(x, bytes),
        range(x, bytes),
        count,
        dtype,
        ReduceOp::Sum,
    )?;
    wait_unless_captured(stream)
}

/// Gathers every rank's `send` into `recv`, in rank order.
///
/// # Errors
///
/// As [`all_reduce`], and [`Error::Collective`] when `recv` does not hold the group's member
/// count times `send`'s elements of the same element type.
pub fn all_gather(send: Operand, recv: Operand) -> Result<()> {
    let (context, ranks) = member("all_gather")?;
    let (count, gathered, dtype) = gather(send, recv, ranks)?;

    let Some(stream) = context.stream() else {
        return Ok(());
    };
    let group = group(&context, "all_gather")?;
    let bytes = dtype.bytes();

    group.communicator().all_gather(
        stream,
        range(send, bytes * count),
        range(recv, bytes * gathered),
        count,
        dtype,
    )?;
    wait_unless_captured(stream)
}

/// Exchanges equal blocks with every rank: block `j` of `send` goes to rank `j`, and block `j`
/// of `recv` is what rank `j` sent here.
///
/// # Errors
///
/// As [`all_reduce`], and [`Error::Collective`] when the buffers are not the group's member count
/// of equal blocks of one element type.
pub fn all_to_all(send: Operand, recv: Operand) -> Result<()> {
    let (context, ranks) = member("all_to_all")?;
    let exchange = exchange(send, recv, ranks)?;

    let Some(stream) = context.stream() else {
        return Ok(());
    };
    group(&context, "all_to_all")?.all_to_all(send, recv, exchange, stream)
}

/// Starts [`all_to_all`] beside the context's stream, which goes on at once.
///
/// The buffers belong to the collective until [`finish`] is given the returned value; what
/// `recv` holds is readable on the context's stream after that.
///
/// # Errors
///
/// As [`all_to_all`].
pub fn start_all_to_all(send: Operand, recv: Operand) -> Result<usize> {
    let (context, ranks) = member("all_to_all")?;
    let exchange = exchange(send, recv, ranks)?;

    let Some(stream) = context.stream() else {
        return Ok(0);
    };
    let group = group(&context, "all_to_all")?;

    group.start(stream, |side| group.all_to_all(send, recv, exchange, side))
}

/// Starts delivering rows `first..first + count` of this rank's `owner_rows` rows to every rank,
/// and theirs here, in place in `gathered`.
///
/// `gathered` is the whole sequence `[ranks * owner_rows, width]`, rank by rank, and this rank's
/// rows of the interval are already in its own block. A row lands where the sequence holds it,
/// so the sequence is whole once the collective of each interval is finished ([`finish`]).
///
/// # Errors
///
/// As [`all_reduce`], and [`Error::Collective`] when `gathered` is not every rank's
/// `owner_rows` rows or the interval is not inside them.
pub fn start_gather_rows(
    gathered: Operand,
    owner_rows: u64,
    first: u64,
    count: u64,
) -> Result<usize> {
    let (context, ranks) = member("gather_rows")?;
    let (_, dtype) = elements(gathered, gathered, "gather_rows")?;
    let width = gathered.shape.last();

    let interval = first.checked_add(count).filter(|end| *end <= owner_rows);
    let whole = ranks
        .checked_mul(owner_rows)
        .and_then(|rows| rows.checked_mul(width));
    if interval.is_none() || whole != Some(gathered.numel()) {
        return Err(misuse(
            "gather_rows",
            format!(
                "rows {first}..+{count} of {owner_rows} per rank in a {} buffer of {ranks} ranks",
                gathered.shape
            ),
        ));
    }

    let Some(stream) = context.stream() else {
        return Ok(0);
    };
    let group = group(&context, "gather_rows")?;

    // One rank's rows of the interval; every rank's lie at the same offset of its own block.
    let elements = usize::try_from(count * width)
        .map_err(|_| misuse("gather_rows", "an element count above usize::MAX"))?;
    let bytes = dtype.bytes() * elements;
    let block = |rank: u64| {
        let at = gathered.offset_elements((rank * owner_rows + first) * width);
        range(gathered.at(at), bytes)
    };

    group.start(stream, |side| {
        group.exchange_rows(block, (elements, dtype), ranks, side)
    })
}

/// Orders the context's stream after the started collective `started`: what it received is
/// readable from here on.
///
/// # Errors
///
/// [`Error::NoContext`]; [`Error::Collective`] when the context carries no group or `started`
/// is not a value a started collective returned; [`Error::Platform`] when the wait cannot be
/// issued.
pub fn finish(started: usize) -> Result<()> {
    let (context, _) = member("finish")?;

    let Some(stream) = context.stream() else {
        return Ok(());
    };
    let group = group(&context, "finish")?;
    let end = group
        .ends
        .get(started)
        .ok_or_else(|| misuse("finish", format!("{started} names no started collective")))?;

    Ok(stream.wait_event(end)?)
}

/// Waits for what the context's stream issued so far, when it is not being captured.
///
/// A module calls this after a launch that waits on its device for the other ranks of the
/// group. Such a launch holds its device until they arrive, and the first use of a kernel takes
/// a lock of the whole process and waits for its device to drain. The ranks of a host are
/// threads of one process, so a rank that went on past such a launch can hold that lock for a
/// kernel that needs its peers, while the peers queue on the lock in front of the same launch. A
/// captured forward runs no kernel and a replay finds every kernel ready, so only an eager
/// forward is held back; the collectives of this module wait in the same way themselves. Under
/// a meta context, and with no context installed, there is nothing to wait for.
///
/// # Errors
///
/// [`Error::Platform`] when the stream cannot be queried or waited for.
pub fn settle() -> Result<()> {
    match context::installed() {
        Some(context) => context.stream().map_or(Ok(()), wait_unless_captured),
        None => Ok(()),
    }
}

/// The thread's context and its group's member count, for the collective `name`.
fn member(name: &'static str) -> Result<(Arc<Context>, u64)> {
    let context = context::current("a collective")?;
    let split = context.state().split;

    match split {
        Some((_, ranks)) => Ok((context, ranks)),
        None => Err(misuse(
            name,
            "a collective runs over a group, and the context carries none",
        )),
    }
}

/// The group a launching context issues the collective `name` through.
fn group(context: &Context, name: &'static str) -> Result<Arc<Group>> {
    let group = context.state().group.clone();

    group.ok_or_else(|| {
        misuse(
            name,
            "the context states a split and holds no communicator to issue through",
        )
    })
}

/// A refused call of `collective`.
fn misuse(collective: &'static str, problem: impl Into<String>) -> Error {
    Error::Collective {
        collective,
        problem: problem.into(),
    }
}

/// The `bytes` bytes `operand` starts at, as the communicator takes a buffer.
fn range(operand: Operand, bytes: usize) -> DeviceRange {
    DeviceRange::new(
        PlatformDevicePtr::from_address(operand.ptr.address()),
        bytes,
    )
}

/// The element count of `send` and the element type NCCL moves it as, refusing buffers of two
/// element types and element types NCCL does not have.
fn elements(
    send: Operand,
    recv: Operand,
    collective: &'static str,
) -> Result<(usize, CollectiveType)> {
    if send.dtype != recv.dtype {
        return Err(misuse(
            collective,
            format!("sends {} and receives {}", send.dtype, recv.dtype),
        ));
    }

    let count = usize::try_from(send.numel())
        .map_err(|_| misuse(collective, "an element count above usize::MAX"))?;
    let dtype = collective_type(send.dtype).ok_or_else(|| {
        misuse(
            collective,
            format!("{} has no NCCL element type", send.dtype),
        )
    })?;

    Ok((count, dtype))
}

/// The element counts of an all-gather over `ranks` members, `send`'s and the `ranks` times as
/// many `recv` must hold, and its element type.
fn gather(send: Operand, recv: Operand, ranks: u64) -> Result<(usize, usize, CollectiveType)> {
    let (count, dtype) = elements(send, recv, "all_gather")?;
    let gathered = usize::try_from(ranks)
        .ok()
        .and_then(|ranks| count.checked_mul(ranks));

    match gathered {
        Some(gathered) if recv.numel() == gathered as u64 => Ok((count, gathered, dtype)),
        _ => Err(misuse(
            "all_gather",
            format!(
                "receives {} elements for {ranks} ranks of {count}",
                recv.numel()
            ),
        )),
    }
}

/// One block's element count, the bytes of each whole buffer and the element type of an
/// all-to-all over `ranks` members, whose buffers both hold `ranks` equal blocks.
fn exchange(send: Operand, recv: Operand, ranks: u64) -> Result<(usize, usize, CollectiveType)> {
    let (count, dtype) = elements(send, recv, "all_to_all")?;
    let ranks = usize::try_from(ranks).unwrap_or(usize::MAX);

    if recv.numel() != send.numel() || !count.is_multiple_of(ranks) {
        return Err(misuse(
            "all_to_all",
            format!(
                "sends {count} elements and receives {}, which are not {ranks} equal blocks",
                recv.numel()
            ),
        ));
    }

    Ok((count / ranks, dtype.bytes() * count, dtype))
}

/// Waits for a collective issued outside a capture, so the forward goes on with none in flight.
///
/// NCCL's kernel spins on its device until every member reaches the collective. The first
/// `cuKernelSetAttribute` of a kernel, which cuBLASLt issues inside a first `cublasLtMatmul` and
/// a native launch inside `cudaFuncSetAttribute`, takes a lock of the whole process and then
/// waits for its device to drain. Every rank of a host is a thread of one process, so a rank
/// that went on past a collective and reached such a call holds that lock for a kernel that
/// needs its peers, while the peers queue on the lock in front of the same collective. The first
/// eager forward of a bucket is where those first calls happen, and the forwards of a group's
/// ranks are not in step, because a rank's rows of each modality are its own.
///
/// A collective inside a capture runs in a replay, by which time every kernel is ready, so it is
/// not waited for.
fn wait_unless_captured(stream: &Stream) -> Result<()> {
    if stream.capture_status()? == CaptureStatus::None {
        stream.synchronize()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use ops::{DType, DevicePtr, Operand, Shape};

    use super::{
        all_gather, all_reduce, all_to_all, finish, settle, split, start_all_to_all,
        start_gather_rows,
    };
    use crate::{Context, Error};

    /// A buffer of `elements` elements of `dtype`; a meta context never reads its address.
    fn buffer(dtype: DType, elements: u64) -> Operand {
        Operand::new(DevicePtr::new(0x4000), dtype, Shape::new(&[elements]))
    }

    /// Whether `result` is the refusal of a misused collective.
    fn misused<T>(result: crate::Result<T>) -> bool {
        matches!(result, Err(Error::Collective { .. }))
    }

    #[test]
    fn a_collective_with_no_context_installed_names_what_is_missing() {
        let refused = all_reduce(buffer(DType::F32, 8)).unwrap_err();
        assert!(matches!(
            refused,
            Error::NoContext {
                operation: "a collective"
            }
        ));
        assert_eq!(split(), None);
        settle().unwrap();
    }

    #[test]
    fn a_collective_is_refused_when_the_context_carries_no_group() {
        let context = Context::meta(None);
        let _guard = context.enter();

        assert_eq!(split(), None);
        assert!(misused(all_reduce(buffer(DType::F32, 8))));
        assert!(misused(finish(0)));
    }

    #[test]
    fn a_meta_context_accepts_the_buffers_its_split_allows_and_issues_nothing() {
        let context = Context::meta(None);
        context.set_group(None, 1, 4).unwrap();
        let _guard = context.enter();

        assert_eq!(split(), Some((1, 4)));
        all_reduce(buffer(DType::BF16, 8)).unwrap();
        all_gather(buffer(DType::F32, 8), buffer(DType::F32, 32)).unwrap();
        all_to_all(buffer(DType::F32, 8), buffer(DType::F32, 8)).unwrap();

        let started = start_all_to_all(buffer(DType::I32, 12), buffer(DType::I32, 12)).unwrap();
        finish(started).unwrap();
        settle().unwrap();
    }

    #[test]
    fn a_meta_context_refuses_buffers_that_do_not_fit_its_split() {
        let context = Context::meta(None);
        context.set_group(None, 1, 4).unwrap();
        let _guard = context.enter();

        // Four ranks of eight elements are thirty-two, of the element type sent.
        assert!(misused(all_gather(
            buffer(DType::F32, 8),
            buffer(DType::F32, 16)
        )));
        assert!(misused(all_gather(
            buffer(DType::F32, 8),
            buffer(DType::BF16, 32)
        )));

        // Six elements are not four equal blocks, and both buffers hold every block.
        assert!(misused(all_to_all(
            buffer(DType::F32, 6),
            buffer(DType::F32, 6)
        )));
        assert!(misused(start_all_to_all(
            buffer(DType::F32, 8),
            buffer(DType::F32, 4)
        )));

        // NCCL has no 16-bit integer element.
        assert!(misused(all_reduce(buffer(DType::I16, 8))));
    }

    #[test]
    fn a_row_gather_takes_an_interval_of_one_rank_s_rows_of_the_whole_sequence() {
        let context = Context::meta(None);
        context.set_group(None, 0, 2).unwrap();
        let _guard = context.enter();

        // Two ranks of five rows, sixteen wide.
        let gathered = Operand::new(DevicePtr::new(0x4000), DType::BF16, Shape::new(&[10, 16]));
        let started = start_gather_rows(gathered, 5, 2, 3).unwrap();
        finish(started).unwrap();

        // Rows 3..7 leave a rank's five rows, and four rows per rank are not this sequence.
        assert!(misused(start_gather_rows(gathered, 5, 3, 4)));
        assert!(misused(start_gather_rows(gathered, 4, 0, 4)));
    }
}

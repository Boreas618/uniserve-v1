//! Collectives over the context's group.
//!
//! A partitioned forward computes one rank's share and meets the other ranks through
//! collectives: a sum over every rank's buffer, a gather of every rank's buffer in rank order,
//! an exchange of equal blocks between every pair of ranks, and one rank's buffer copied to
//! every rank ([`broadcast`]), as a pipeline stage hands its rows to the stages after it. A
//! module reads its place in the group with [`split`] and calls the collective on its tensors;
//! which communicator and stream carry it is the context's.
//!
//! A collective launch holds its device until every member of the group reached it. The members
//! of a host are threads of one process, and a member's thread can need its process while a
//! peer's device waits in such a launch: the first use of a kernel takes a lock of the whole
//! process and waits for the work in flight to drain, and a pipeline stage's thread issues a
//! whole stage of layers while the next stage's device waits in their broadcast (an eager
//! pipeline forward stalled there). A group that states how its members on the host meet
//! ([`Group::with_meet`]) therefore meets before each such launch issued outside a capture, so
//! no member's device waits on a member whose thread has not reached the launch. A captured
//! launch holds no device, and a replay finds every kernel ready.
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
//!
//! The ranks of an expert group meet through the **expert exchange** ([`expert_exchange`]): a
//! kernel package's transport moves each route to the rank that holds its expert and each
//! token's sums back. The transport is the context's, beside the group
//! ([`Context::set_exchange`]), so a module states the exchange's tensors and never the handle.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use platform::nccl::{Communicator, DataType, ReduceOp, group as nccl_group};
use platform::{CaptureStatus, DeviceRange, Event, EventFlags, Stream};
use tensor::{DType, Tensor};

use crate::Operand;
use crate::context::{self, Context};
use crate::moe::exchange::Descriptor as ExchangeDescriptor;
use crate::moe::{ExchangeArgs, ExpertExchange};
use crate::{Error, Result, launch, storage};

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
/// which the one worker thread of a device already keeps; the lock states it in the type, so
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
    /// How the group's members on this host meet before each collective launch outside a
    /// capture ([`Group::with_meet`]); `None` for a group whose members need not.
    meet: Option<Meet>,
}

/// A host rendezvous of the members of a group on one host: it returns once each of them called
/// it.
pub type Meet = Arc<dyn Fn() -> Result<()> + Send + Sync>;

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
            meet: None,
        })
    }

    /// The group whose members on this host call `meet` together before each collective launch
    /// issued outside a capture that waits on its device for the other members (module
    /// documentation): every launch of a group but a finish. Every member issues the group's
    /// collectives in the same order and captures the same ones, so the `n`-th meeting of every
    /// member is the same launch's.
    #[must_use]
    pub fn with_meet(self, meet: Meet) -> Self {
        Self {
            meet: Some(meet),
            ..self
        }
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
        send: &Operand,
        recv: &Operand,
        (block, bytes, dtype): (usize, usize, DataType),
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
        (elements, dtype): (usize, DataType),
        ranks: u64,
        stream: &Stream,
    ) -> Result<()> {
        {
            let communicator = self.communicator();
            let rank = communicator.rank();

            nccl_group(|| {
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

/// Runs `operation` with the forward's collectives and [`split`] along axis `axis` of the
/// context ([`Context::set_axes`]), and the axis addressed before it again once `operation`
/// returns, whatever it returns.
///
/// A forward partitioned along several axes (a sequence split inside a tensor split) addresses
/// each collective's axis this way; outside any, its collectives go along the first. A
/// collective started along an axis is finished along it.
///
/// # Errors
///
/// [`Error::NoContext`], or [`Error::Collective`] when the context states no axis `axis`.
pub fn along<T>(axis: usize, operation: impl FnOnce() -> T) -> Result<T> {
    let context = context::current("a collective")?;
    let previous = {
        let mut state = context.state();
        let (group, split) = state.axes.get(axis).cloned().ok_or_else(|| {
            misuse(
                "along",
                format!(
                    "axis {axis} of a forward partitioned along {} axes",
                    state.axes.len()
                ),
            )
        })?;
        (
            std::mem::replace(&mut state.group, group),
            std::mem::replace(&mut state.split, split),
        )
    };

    let result = operation();

    let mut state = context.state();
    (state.group, state.split) = previous;
    Ok(result)
}

/// Sums every rank's `x` into every rank's `x`, in place.
///
/// `x` has the same shape and element type on every rank. Like every buffer of a collective it
/// is a contiguous tensor of this backend, meta under a meta context.
///
/// # Errors
///
/// [`Error::NoContext`]; [`Error::Collective`] when the context carries no group, `x` is not
/// contiguous or NCCL has no element type for it; [`Error::ForeignTensor`] for a CPU tensor;
/// [`Error::Platform`] when the collective cannot be issued.
pub fn all_reduce(x: &Tensor) -> Result<()> {
    let (context, _) = member("all_reduce")?;
    let x = buffer(x, "all_reduce")?;
    let (count, dtype) = elements(&x, &x, "all_reduce")?;

    let Some(stream) = context.stream() else {
        return Ok(());
    };
    let group = group(&context, "all_reduce")?;
    meet(&group, stream)?;
    let bytes = byte_count(&x, count, "all_reduce")?;

    group.communicator().all_reduce(
        stream,
        range(&x, bytes),
        range(&x, bytes),
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
pub fn all_gather(send: &Tensor, recv: &Tensor) -> Result<()> {
    let (context, ranks) = member("all_gather")?;
    let (send, recv) = (buffer(send, "all_gather")?, buffer(recv, "all_gather")?);
    let (count, gathered, dtype) = gather(&send, &recv, ranks)?;

    let Some(stream) = context.stream() else {
        return Ok(());
    };
    let group = group(&context, "all_gather")?;
    meet(&group, stream)?;
    group.communicator().all_gather(
        stream,
        range(&send, byte_count(&send, count, "all_gather")?),
        range(&recv, byte_count(&recv, gathered, "all_gather")?),
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
pub fn all_to_all(send: &Tensor, recv: &Tensor) -> Result<()> {
    let (context, ranks) = member("all_to_all")?;
    let (send, recv) = (buffer(send, "all_to_all")?, buffer(recv, "all_to_all")?);
    let exchange = exchange(&send, &recv, ranks)?;

    let Some(stream) = context.stream() else {
        return Ok(());
    };
    let group = group(&context, "all_to_all")?;
    meet(&group, stream)?;
    group.all_to_all(&send, &recv, exchange, stream)
}

/// Starts [`all_to_all`] beside the context's stream, which goes on at once.
///
/// The buffers belong to the collective until [`finish`] is given the returned value: the
/// caller keeps both tensors until then, and what `recv` holds is readable on the context's
/// stream after that.
///
/// # Errors
///
/// As [`all_to_all`].
pub fn start_all_to_all(send: &Tensor, recv: &Tensor) -> Result<usize> {
    let (context, ranks) = member("all_to_all")?;
    let (send, recv) = (buffer(send, "all_to_all")?, buffer(recv, "all_to_all")?);
    let exchange = exchange(&send, &recv, ranks)?;

    let Some(stream) = context.stream() else {
        return Ok(0);
    };
    let group = group(&context, "all_to_all")?;
    meet(&group, stream)?;

    group.start(stream, |side| {
        group.all_to_all(&send, &recv, exchange, side)
    })
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
    gathered: &Tensor,
    owner_rows: u64,
    first: u64,
    count: u64,
) -> Result<usize> {
    let (context, ranks) = member("gather_rows")?;
    let gathered = buffer(gathered, "gather_rows")?;
    let (_, dtype) = elements(&gathered, &gathered, "gather_rows")?;
    let width = gathered.last();

    let interval = first.checked_add(count).filter(|end| *end <= owner_rows);
    let whole = ranks
        .checked_mul(owner_rows)
        .and_then(|rows| rows.checked_mul(width));
    if interval.is_none() || whole != Some(gathered.numel()) {
        return Err(misuse(
            "gather_rows",
            format!(
                "rows {first}..+{count} of {owner_rows} per rank in a {:?} buffer of {ranks} ranks",
                gathered.dims()
            ),
        ));
    }

    let Some(stream) = context.stream() else {
        return Ok(0);
    };
    let group = group(&context, "gather_rows")?;
    meet(&group, stream)?;

    // One rank's rows of the interval; every rank's lie at the same offset of its own block.
    let elements = usize::try_from(count * width)
        .map_err(|_| misuse("gather_rows", "an element count above usize::MAX"))?;
    let bytes = byte_count(&gathered, elements, "gather_rows")?;
    let block = |rank: u64| {
        let at = gathered.offset_elements((rank * owner_rows + first) * width);
        DeviceRange::new(at, bytes)
    };

    group.start(stream, |side| {
        group.exchange_rows(block, (elements, dtype), ranks, side)
    })
}

/// Rank `root`'s `x` on every rank of the group: on `root`, `x` itself, which the collective
/// reads in place; on every other rank a new tensor of `x`'s shape and element type holding
/// `root`'s elements. Every rank passes a tensor of the same shape and element type, which
/// only `root` reads.
///
/// # Errors
///
/// As [`all_reduce`], and [`Error::Collective`] when `root` is no rank of the group.
pub fn broadcast(x: &Tensor, root: u64) -> Result<Tensor> {
    let (context, ranks) = member("broadcast")?;
    let rank = context.state().split.map_or(0, |(rank, _)| rank);
    if root >= ranks {
        return Err(misuse(
            "broadcast",
            format!("rank {root} of a group of {ranks}"),
        ));
    }

    // The root's tensor is the one every rank ends with, so it is the root's own output too.
    let result = if rank == root {
        x.clone()
    } else {
        crate::empty(x.shape(), x.dtype())?
    };
    let operand = buffer(&result, "broadcast")?;
    let (count, dtype) = elements(&operand, &operand, "broadcast")?;

    let Some(stream) = context.stream() else {
        return Ok(result);
    };
    let group = group(&context, "broadcast")?;
    meet(&group, stream)?;
    let bytes = byte_count(&operand, count, "broadcast")?;

    group.communicator().broadcast(
        stream,
        range(&operand, bytes),
        range(&operand, bytes),
        count,
        dtype,
        nccl_rank(root),
    )?;
    wait_unless_captured(stream)?;
    Ok(result)
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

/// One launch of the expert exchange `descriptor` over the context's transport.
///
/// `args` states the launch's tensors and leaves its transport unset: the context's is used
/// ([`Context::set_exchange`]). The launch is checked, resolved and recorded as every launch is
/// ([`launch()`]), so under a meta context it checks its tensors and states its key with no
/// transport set.
///
/// An exchange launch waits on its device for the same launch of every rank of the expert
/// group. Under a context that launches, the context's `meet` is therefore called first, and the
/// stream is waited for afterwards unless it is being captured ([`settle`]).
///
/// # Errors
///
/// [`Error::NoContext`]; [`Error::Collective`] when a context that launches holds no transport;
/// what `meet` returns; [`Error::Launch`] as [`launch()`]; [`Error::Platform`] when the stream
/// cannot be waited for.
pub fn expert_exchange(descriptor: &ExchangeDescriptor, args: ExchangeArgs) -> Result<()> {
    let context = context::current("an expert exchange")?;
    let (transport, meet) = {
        let state = context.state();
        (state.exchange, state.meet.clone())
    };

    if !context.is_meta() {
        if transport.is_none() {
            return Err(misuse(
                "expert_exchange",
                "an exchange runs over the expert group's transport, and the context holds none",
            ));
        }
        if let Some(meet) = meet {
            meet()?;
        }
    }

    launch::<ExpertExchange>(descriptor, ExchangeArgs { transport, ..args })?;
    settle()
}

/// Meets the other members of `group` on the host before a collective launch on `stream`, when
/// the group states a meeting ([`Group::with_meet`]) and `stream` is not being captured.
fn meet(group: &Group, stream: &Stream) -> Result<()> {
    match &group.meet {
        Some(meet) if stream.capture_status()? == CaptureStatus::None => meet(),
        _ => Ok(()),
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

/// `rank`, a rank of a group whose member count NCCL reported as an `int`, as NCCL names it.
fn nccl_rank(rank: u64) -> i32 {
    i32::try_from(rank).unwrap_or(i32::MAX)
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

/// The buffer of the collective `name` that `tensor` is: its kernel argument, from whose
/// address the collective moves the tensor's elements.
///
/// A collective moves a run of bytes, so the tensor is contiguous; a view with a stride would
/// send the elements between its own.
fn buffer(tensor: &Tensor, name: &'static str) -> Result<Operand> {
    if !tensor.is_contiguous() {
        return Err(misuse(
            name,
            format!(
                "a collective moves a contiguous buffer, and a tensor of shape {:?} is a strided \
                 view",
                tensor.shape()
            ),
        ));
    }

    storage::operand(tensor)
}

/// A refused call of `collective`.
fn misuse(collective: &'static str, problem: impl Into<String>) -> Error {
    Error::Collective {
        collective,
        problem: problem.into(),
    }
}

/// The bytes `count` elements of `operand`'s element type occupy ([`DType::storage_bytes`]).
fn byte_count(operand: &Operand, count: usize, collective: &'static str) -> Result<usize> {
    operand
        .dtype
        .storage_bytes(count)
        .ok_or_else(|| misuse(collective, "a byte count above usize::MAX"))
}

/// The `bytes` bytes `operand` starts at, as the communicator takes a buffer.
fn range(operand: &Operand, bytes: usize) -> DeviceRange {
    DeviceRange::new(operand.ptr, bytes)
}

/// The element type NCCL moves `dtype` as, when NCCL has one: the one conversion of an element
/// type into the communicator's.
///
/// A sub-byte type has no collective counterpart: it travels inside a program's own encoded
/// layout, never as a buffer a collective reads.
fn nccl_type(dtype: DType) -> Option<DataType> {
    Some(match dtype {
        DType::U8 => DataType::Uint8,
        DType::I32 => DataType::Int32,
        DType::U32 => DataType::Uint32,
        DType::I64 => DataType::Int64,
        DType::F16 => DataType::Float16,
        DType::F32 => DataType::Float32,
        DType::F64 => DataType::Float64,
        DType::BF16 => DataType::Bfloat16,
        DType::F8E4M3 => DataType::Float8e4m3,
        // NCCL has no 16-bit integer reduction: a silent fallback would be a wrong answer, so a
        // PCM sample travels as bytes and never as a collective's element.
        DType::I16 => return None,
        // Neither a packed element nor one the tensor library adds is a collective's element.
        _ => return None,
    })
}

/// The element count of `send` and the element type NCCL moves it as, refusing buffers of two
/// element types and element types NCCL does not have.
fn elements(send: &Operand, recv: &Operand, collective: &'static str) -> Result<(usize, DataType)> {
    if send.dtype != recv.dtype {
        return Err(misuse(
            collective,
            format!(
                "sends {} and receives {}",
                crate::abi::name(send.dtype),
                crate::abi::name(recv.dtype)
            ),
        ));
    }

    let count = usize::try_from(send.numel())
        .map_err(|_| misuse(collective, "an element count above usize::MAX"))?;
    let dtype = nccl_type(send.dtype).ok_or_else(|| {
        misuse(
            collective,
            format!("{} has no NCCL element type", crate::abi::name(send.dtype)),
        )
    })?;

    Ok((count, dtype))
}

/// The element counts of an all-gather over `ranks` members, `send`'s and the `ranks` times as
/// many `recv` must hold, and its element type.
fn gather(send: &Operand, recv: &Operand, ranks: u64) -> Result<(usize, usize, DataType)> {
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
fn exchange(send: &Operand, recv: &Operand, ranks: u64) -> Result<(usize, usize, DataType)> {
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

    Ok((count / ranks, byte_count(send, count, "all_to_all")?, dtype))
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
    use tensor::{DType, Device, Tensor};

    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::moe::ExchangeArgs;
    use crate::moe::exchange::{Descriptor, Mode};

    use super::{
        all_gather, all_reduce, all_to_all, broadcast, expert_exchange, finish, settle, split,
        start_all_to_all, start_gather_rows,
    };
    use crate::{Context, Error, meta, operand, retained};

    /// A meta buffer of `elements` elements of `dtype`: no context reads its address.
    fn buffer(dtype: DType, elements: usize) -> Tensor {
        retained(0x4000, elements, dtype, &meta()).unwrap()
    }

    /// Whether `result` is the refusal of a misused collective.
    fn misused<T>(result: crate::Result<T>) -> bool {
        matches!(result, Err(Error::Collective { .. }))
    }

    #[test]
    fn a_collective_with_no_context_installed_names_what_is_missing() {
        let refused = all_reduce(&buffer(DType::F32, 8)).unwrap_err();
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
        assert!(misused(all_reduce(&buffer(DType::F32, 8))));
        assert!(misused(finish(0)));
    }

    #[test]
    fn a_meta_context_accepts_the_buffers_its_split_allows_and_issues_nothing() {
        let context = Context::meta(None);
        context.set_group(None, 1, 4).unwrap();
        let _guard = context.enter();

        assert_eq!(split(), Some((1, 4)));
        all_reduce(&buffer(DType::BF16, 8)).unwrap();
        all_gather(&buffer(DType::F32, 8), &buffer(DType::F32, 32)).unwrap();
        all_to_all(&buffer(DType::F32, 8), &buffer(DType::F32, 8)).unwrap();

        let started = start_all_to_all(&buffer(DType::I32, 12), &buffer(DType::I32, 12)).unwrap();
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
            &buffer(DType::F32, 8),
            &buffer(DType::F32, 16)
        )));
        assert!(misused(all_gather(
            &buffer(DType::F32, 8),
            &buffer(DType::BF16, 32)
        )));

        // Six elements are not four equal blocks, and both buffers hold every block.
        assert!(misused(all_to_all(
            &buffer(DType::F32, 6),
            &buffer(DType::F32, 6)
        )));
        assert!(misused(start_all_to_all(
            &buffer(DType::F32, 8),
            &buffer(DType::F32, 4)
        )));

        // NCCL has no 16-bit integer element.
        assert!(misused(all_reduce(&buffer(DType::I16, 8))));
    }

    #[test]
    fn a_meta_context_broadcasts_the_root_s_tensor_to_every_rank() {
        let context = Context::meta(None);
        context.set_group(None, 1, 3).unwrap();
        let _guard = context.enter();

        // The root ends with its own tensor, every other rank with one of the root's shape.
        let rows = buffer(DType::BF16, 12);
        assert_eq!(
            operand(&broadcast(&rows, 1).unwrap()).unwrap().ptr,
            operand(&rows).unwrap().ptr
        );
        let received = broadcast(&rows, 2).unwrap();
        assert_eq!(
            (received.dims(), received.dtype()),
            (rows.dims(), DType::BF16)
        );

        // A root is a rank of the group.
        assert!(misused(broadcast(&rows, 3)));
    }

    #[test]
    fn a_row_gather_takes_an_interval_of_one_rank_s_rows_of_the_whole_sequence() {
        let context = Context::meta(None);
        context.set_group(None, 0, 2).unwrap();
        let _guard = context.enter();

        // Two ranks of five rows, sixteen wide.
        let gathered = retained(0x4000, (10, 16), DType::BF16, &meta()).unwrap();
        let started = start_gather_rows(&gathered, 5, 2, 3).unwrap();
        finish(started).unwrap();

        // Rows 3..7 leave a rank's five rows, and four rows per rank are not this sequence.
        assert!(misused(start_gather_rows(&gathered, 5, 3, 4)));
        assert!(misused(start_gather_rows(&gathered, 4, 0, 4)));
    }

    /// The dispatch of four rows of sixty-four elements routed to two experts each, by a rank of
    /// a two-rank expert group.
    fn dispatch(ranks: u32) -> (Descriptor, ExchangeArgs) {
        let descriptor = Descriptor {
            mode: Mode::Dispatch,
            hidden: 64,
            intermediate: 128,
            top: 2,
            experts: 8,
            max_tokens: 16,
            ranks: 2,
            attention: 0,
            dtype: DType::BF16,
        };
        let tensor = |shape: (usize, usize), dtype| {
            operand(&retained(0x4000, shape, dtype, &meta()).unwrap()).unwrap()
        };
        let args = ExchangeArgs {
            transport: None,
            tokens: 4,
            state: tensor((4, 64), DType::BF16),
            ids: tensor((4, 2), DType::U32),
            weights: tensor((4, 2), DType::F32),
            live: None,
            out: tensor((4, 64), DType::BF16),
            fused: None,
            ranks,
        };

        (descriptor, args)
    }

    #[test]
    fn an_expert_exchange_needs_a_context() {
        let (descriptor, args) = dispatch(2);

        assert!(matches!(
            expert_exchange(&descriptor, args),
            Err(Error::NoContext {
                operation: "an expert exchange"
            })
        ));
    }

    #[test]
    fn a_meta_context_checks_an_expert_exchange_and_states_its_key_without_a_transport() {
        let context = Context::meta(None);
        context.record_launches();
        let _guard = context.enter();

        let (descriptor, args) = dispatch(2);
        expert_exchange(&descriptor, args).unwrap();

        // A launch over another group size than the kernel is compiled for is refused by the
        // operator's own check, and a refused call is not a launch.
        let (descriptor, wider) = dispatch(4);
        assert!(matches!(
            expert_exchange(&descriptor, wider),
            Err(Error::Launch {
                operator: "expert_exchange",
                ..
            })
        ));

        let launches: Vec<_> = context
            .take_launches()
            .into_iter()
            .map(|(operator, key, path, _)| (operator, key, path))
            .collect();
        assert_eq!(
            launches,
            [(
                "expert_exchange",
                "mode_dispatch_hidden_64_intermediate_128_top_2_experts_8_max_tokens_16_ranks_2_\
                 attention_0_dtype_bf16"
                    .to_owned(),
                String::new()
            )]
        );
    }

    #[test]
    fn a_meta_context_issues_no_exchange_and_meets_no_rank() {
        let met = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&met);

        let context = Context::meta(None);
        context.set_exchange(
            None,
            Some(Arc::new(move || {
                counted.fetch_add(1, Ordering::Relaxed);
                Ok(())
            })),
        );
        let _guard = context.enter();

        let (descriptor, args) = dispatch(2);
        expert_exchange(&descriptor, args).unwrap();
        assert_eq!(met.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn a_collective_moves_a_contiguous_tensor_of_this_backend() {
        let context = Context::meta(None);
        context.set_group(None, 0, 2).unwrap();
        let _guard = context.enter();

        // Every other column of a matrix is not a run of bytes.
        let columns = retained(0x4000, (4, 8), DType::F32, &meta()).unwrap();
        let strided = columns.transpose(0, 1).unwrap();
        assert!(misused(all_reduce(&strided)));

        let host = Tensor::zeros(8, DType::F32, &Device::Cpu).unwrap();
        assert!(matches!(
            all_reduce(&host),
            Err(Error::ForeignTensor { .. })
        ));
    }
}

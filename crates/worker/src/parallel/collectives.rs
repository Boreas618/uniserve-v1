//! Stream-ordered NCCL collectives and registered window lifetime.

use std::sync::{Arc, PoisonError};

use tvm_ffi::{Any, AnyView};

use indexmap::IndexMap;
use uniserve_cuda::nccl::{self, Communicator, CtaPolicy, DataType, ReduceOp, UniqueId};
use uniserve_cuda::{DeviceId, Event, EventFlags, Stream};

use crate::{CUDAStream, Error, Result};

/// An NCCL communicator bound to one computation stream.
///
/// Synchronous operations run on the computation stream. Asynchronous
/// operations run on a transfer stream forked from it, which shares its SM
/// partition; consumers join them by event. Registered window backing lives
/// through communicator retirement, independently of the execution contexts
/// and graphs borrowing it.
pub struct StreamCommunicator<T> {
    device: DeviceId,
    resources: Option<Resources<T>>,
}

struct Resources<T> {
    communicator: Communicator,
    // The owner's stream, which synchronous operations run on.
    computation: Stream,
    transfer: CUDAStream,
    // The latest asynchronous operation not yet joined on the computation
    // stream. Synchronous operations order themselves after it.
    pending: Option<Arc<Event>>,
    // Registered `(address, bytes)` windows and the owners of their memory.
    windows: IndexMap<(u64, usize), T>,
}

impl<T> StreamCommunicator<T> {
    /// Every group member calls this once, with the same distributed unique ID.
    ///
    /// Communicators of full-device streams use NCCL's copy-engine algorithms
    /// for registered windows; SM-partitioned streams use its CTA algorithms.
    ///
    /// The communicator shares `owner`'s stream and keeps submitting to it
    /// until [`Self::close`].
    pub fn new(owner: &CUDAStream, size: i32, rank: i32, id: &UniqueId) -> Result<Self> {
        let device = owner.device();
        let computation = owner.stream()?.clone();

        let policy = if owner.full_device() {
            CtaPolicy::Zero
        } else {
            CtaPolicy::Default
        };
        let communicator = Communicator::init(device, size, rank, id, policy)?;

        let transfer = match owner.fork() {
            Ok(stream) => stream,
            Err(error) => {
                let _ = communicator.abort();
                return Err(error);
            }
        };

        Ok(Self {
            device,
            resources: Some(Resources {
                communicator,
                computation,
                transfer,
                pending: None,
                windows: IndexMap::new(),
            }),
        })
    }

    /// The device the communicator's streams run on.
    pub fn device(&self) -> DeviceId {
        self.device
    }

    /// Members of the communicator.
    pub fn size(&self) -> Result<usize> {
        Ok(self.resources()?.communicator.size() as usize)
    }

    /// This member's backend rank.
    pub fn rank(&self) -> Result<i32> {
        Ok(self.resources()?.communicator.rank())
    }

    pub fn transfer_stream(&self) -> Result<usize> {
        self.resources()?.transfer.handle()
    }

    fn resources(&self) -> Result<&Resources<T>> {
        self.resources
            .as_ref()
            .ok_or(Error::State("computation collective is closed"))
    }

    fn resources_mut(&mut self) -> Result<&mut Resources<T>> {
        self.resources
            .as_mut()
            .ok_or(Error::State("computation collective is closed"))
    }

    /// The communicator and the stream an operation is submitted on.
    fn submission(&mut self, asynchronous: bool) -> Result<(&mut Communicator, &Stream)> {
        let resources = self.resources_mut()?;
        let stream = if asynchronous {
            resources.transfer.stream()?
        } else {
            // A projection can invoke another collective before consuming its
            // gathered tail. Keep both streams on the same communicator order.
            if let Some(pending) = resources.pending.take() {
                resources.computation.wait_event(&pending)?;
            }
            &resources.computation
        };
        Ok((&mut resources.communicator, stream))
    }

    fn start(&mut self, call: impl FnOnce(&mut Self) -> Result<()>) -> Result<Arc<Event>> {
        let resources = self.resources_mut()?;
        resources.transfer.wait(&resources.computation)?;
        let completed = Arc::new(Event::new(EventFlags::default()));
        let result = call(self);

        let resources = self.resources_mut()?;
        completed.record(resources.transfer.stream()?)?;
        if let Err(error) = result {
            // Join partial submissions before the caller can retire inputs.
            resources.computation.wait_event(&completed)?;
            return Err(error);
        }
        resources.pending = Some(Arc::clone(&completed));
        Ok(completed)
    }

    /// Orders `consumer` after an asynchronous operation's completion.
    pub fn join(&mut self, event: &Arc<Event>, consumer: &Stream) -> Result<()> {
        consumer.wait_event(event)?;

        let resources = self.resources_mut()?;
        if consumer.raw() == resources.computation.raw()
            && resources
                .pending
                .as_ref()
                .is_some_and(|pending| Arc::ptr_eq(pending, event))
        {
            // A completed eager join must not leak an event dependency into
            // a subsequent, independent graph capture.
            resources.pending = None;
        }
        Ok(())
    }

    /// Gathers `count` elements from every rank into `output`, rank-major.
    /// Asynchronous gathers return their completion event.
    ///
    /// # Safety
    /// Input and output spans must hold the declared elements on this device
    /// and remain alive through the submitted operation and graph replays.
    pub unsafe fn all_gather(
        &mut self,
        output: u64,
        input: u64,
        count: usize,
        dtype: DataType,
        asynchronous: bool,
    ) -> Result<Option<Arc<Event>>> {
        let submit = |owner: &mut Self| {
            let (communicator, stream) = owner.submission(asynchronous)?;
            // SAFETY: forwarded from the caller.
            unsafe { communicator.all_gather(input, output, count, dtype, stream) }?;
            Ok(())
        };

        if asynchronous {
            self.start(submit).map(Some)
        } else {
            submit(self).map(|()| None)
        }
    }

    /// Exchanges `count`-element blocks with every rank. Asynchronous
    /// exchanges return their completion event.
    ///
    /// # Safety
    /// Each span holds `count` elements per rank and outlives all device uses.
    pub unsafe fn all_to_all(
        &mut self,
        output: u64,
        input: u64,
        count: usize,
        dtype: DataType,
        asynchronous: bool,
    ) -> Result<Option<Arc<Event>>> {
        let submit = |owner: &mut Self| {
            let (communicator, stream) = owner.submission(asynchronous)?;
            // SAFETY: forwarded from the caller.
            unsafe { communicator.all_to_all(input, output, count, dtype, stream) }?;
            Ok(())
        };

        if asynchronous {
            self.start(submit).map(Some)
        } else {
            submit(self).map(|()| None)
        }
    }

    /// # Safety
    /// The in-place tensor holds `count` elements and outlives device uses.
    pub unsafe fn all_reduce(
        &mut self,
        data: u64,
        count: usize,
        dtype: DataType,
        op: ReduceOp,
    ) -> Result<()> {
        let (communicator, stream) = self.submission(false)?;
        // SAFETY: forwarded from the caller.
        unsafe { communicator.all_reduce(data, data, count, dtype, op, stream) }?;
        Ok(())
    }

    /// # Safety
    /// The in-place tensor holds `count` elements and outlives device uses.
    pub unsafe fn broadcast(
        &mut self,
        data: u64,
        count: usize,
        dtype: DataType,
        root: i32,
    ) -> Result<()> {
        let (communicator, stream) = self.submission(false)?;
        // SAFETY: forwarded from the caller.
        unsafe { communicator.broadcast(data, data, count, dtype, root, stream) }?;
        Ok(())
    }

    /// Sums `count` elements per rank and leaves this rank's block in `output`.
    ///
    /// # Safety
    /// Input holds `count` elements per rank and output holds `count` elements.
    /// Both remain allocated through device reads and writes.
    pub unsafe fn reduce_scatter(
        &mut self,
        output: u64,
        input: u64,
        count: usize,
        dtype: DataType,
    ) -> Result<()> {
        let (communicator, stream) = self.submission(false)?;
        // SAFETY: forwarded from the caller.
        unsafe { communicator.reduce_scatter(input, output, count, dtype, ReduceOp::Sum, stream) }?;
        Ok(())
    }

    /// # Safety
    /// The span holds `count` elements until its transfer and graph uses finish.
    pub unsafe fn send(
        &mut self,
        data: u64,
        count: usize,
        dtype: DataType,
        peer: i32,
    ) -> Result<()> {
        let (communicator, stream) = self.submission(false)?;
        // SAFETY: forwarded from the caller.
        unsafe { communicator.send(data, count, dtype, peer, stream) }?;
        Ok(())
    }

    /// # Safety
    /// The destination holds `count` elements until its transfer and graph uses finish.
    pub unsafe fn recv(
        &mut self,
        data: u64,
        count: usize,
        dtype: DataType,
        peer: i32,
    ) -> Result<()> {
        let (communicator, stream) = self.submission(false)?;
        // SAFETY: forwarded from the caller.
        unsafe { communicator.recv(data, count, dtype, peer, stream) }?;
        Ok(())
    }

    /// Group matching point-to-point operations into one NCCL submission.
    /// The group is ended even when `call` fails.
    pub fn grouped<R>(&mut self, call: impl FnOnce(&mut Self) -> Result<R>) -> Result<R> {
        let mut outcome = None;
        nccl::group(|| {
            outcome = Some(call(self));
            Ok(())
        })?;
        // `group` runs the submission whenever the group starts.
        outcome.unwrap_or_else(|| {
            Err(Error::Invariant(
                "grouped collective submission did not run".into(),
            ))
        })
    }

    /// Retain one symmetric allocation through all contexts sharing this stream.
    ///
    /// # Safety
    /// `backing` keeps the VMM allocation `[data, data + bytes)` live; every rank
    /// registers matching windows in the same order, outside graph capture.
    pub unsafe fn register(&mut self, data: u64, bytes: usize, backing: T) -> Result<()> {
        let resources = self.resources_mut()?;
        if resources.windows.contains_key(&(data, bytes)) {
            return Ok(());
        }

        // SAFETY: forwarded from the caller; `backing` is retained below until
        // the communicator is destroyed or its resources are leaked.
        unsafe { resources.communicator.register_window(data, bytes) }?;
        resources.windows.insert((data, bytes), backing);
        Ok(())
    }

    /// Close collectively after all borrowed graph and tensor accesses retire.
    /// Aborted execution retains handles, mappings and streams until process exit.
    #[allow(clippy::mem_forget)]
    pub fn close(&mut self, aborted: bool) -> Result<()> {
        let Some(resources) = self.resources.take() else {
            return Ok(());
        };
        let Resources {
            communicator,
            computation: _,
            mut transfer,
            pending: _,
            windows,
        } = resources;

        // Without a completed destroy, the communicator, its window backing,
        // its transfer stream and its context reference stay alive until
        // process exit.
        let retain = |transfer: CUDAStream, windows: IndexMap<(u64, usize), T>| {
            std::mem::forget(transfer);
            std::mem::forget(windows);
        };

        if aborted {
            std::mem::forget(communicator);
            retain(transfer, windows);
            return Ok(());
        }

        if let Err(error) = transfer.synchronize() {
            std::mem::forget(communicator);
            retain(transfer, windows);
            return Err(error);
        }
        // Deregisters every window, then destroys the communicator. A failure
        // leaks the communicator's remaining resources.
        if let Err(error) = communicator.destroy() {
            retain(transfer, windows);
            return Err(error.into());
        }
        if let Err(error) = transfer.close(false) {
            retain(transfer, windows);
            return Err(error);
        }
        Ok(())
    }

    /// Visit backing references held by a foreign-language resource owner.
    pub fn backings(&self) -> impl Iterator<Item = &T> {
        self.resources
            .iter()
            .flat_map(|resources| resources.windows.values())
    }
}

impl<T> Drop for StreamCommunicator<T> {
    fn drop(&mut self) {
        // Implicit destruction cannot enter a collective that peers may never
        // reach. Normal callers explicitly close after retiring graph readers.
        let _ = self.close(true);
    }
}

/// One mesh group a binding's numerical calls communicate over.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectiveGroup {
    /// The communicator's name: its mesh axes joined by `.` (`tp`).
    pub name: String,
    /// Members as indices of the worker's host group, in logical order.
    pub members: Vec<usize>,
    /// Members' process-world ranks, in the same order; numerical code
    /// names roots and peers by them.
    pub ranks: Vec<usize>,
}

struct GroupComm {
    group: CollectiveGroup,
    comm: Arc<std::sync::Mutex<StreamCommunicator<()>>>,
}

/// NCCL communicators bound to one binding's stream, one per mesh group its
/// numerical calls use, and their TVM-FFI collective tables.
///
/// Communicator ranks follow ascending process-world rank, as the backend
/// order of `uniserve_models.distributed.Communicator` assumes; collectives
/// enqueue on the binding's stream (asynchronous ones on its transfer fork),
/// so they are ordered with the binding's computation and captured with it.
pub struct StreamCollectives {
    groups: Vec<GroupComm>,
}

impl StreamCollectives {
    /// Create the communicators of `groups` for `owner`'s stream.
    ///
    /// Collective over every member of each group, in `groups` order, which
    /// must be identical on every rank: each group's first backend member
    /// creates its NCCL unique id and sends it over `host`. `rank` is this
    /// rank's index in `host`.
    ///
    /// The communicators share `owner`'s stream and submit to it until
    /// [`Self::close`].
    pub fn bind(
        owner: &CUDAStream,
        host: &super::host::HostGroup,
        groups: &[CollectiveGroup],
    ) -> Result<Self> {
        let mut bound = Self { groups: Vec::new() };
        for group in groups {
            if group.members.len() < 2 {
                continue;
            }
            if !group.members.contains(&host.rank()) || group.members.len() != group.ranks.len() {
                return Err(Error::Invalid(format!(
                    "collective group {} must contain this rank once per member",
                    group.name
                )));
            }
            // Backend order: ascending world rank.
            let mut order: Vec<(usize, usize)> = group
                .ranks
                .iter()
                .copied()
                .zip(group.members.iter().copied())
                .collect();
            order.sort_unstable();
            let members: Vec<usize> = order.iter().map(|&(_, member)| member).collect();
            let rank = members
                .iter()
                .position(|&member| member == host.rank())
                .ok_or_else(|| Error::Invariant("a collective member left its group".into()))?;
            let subgroup = host.subgroup(&members)?;
            let id = if rank == 0 {
                let id = UniqueId::new()?;
                subgroup.broadcast_bytes(Some(&id.0), 0)?;
                id
            } else {
                let bytes = subgroup.broadcast_bytes(None, 0)?;
                UniqueId(
                    bytes
                        .as_slice()
                        .try_into()
                        .map_err(|_| Error::Transport("an NCCL unique id has 128 bytes".into()))?,
                )
            };
            // SAFETY: forwarded from the caller.
            let comm = StreamCommunicator::new(owner, members.len() as i32, rank as i32, &id)?;
            bound.groups.push(GroupComm {
                group: group.clone(),
                comm: Arc::new(std::sync::Mutex::new(comm)),
            });
        }
        Ok(bound)
    }

    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// Every group's collective table, by communicator name.
    pub fn tables(&self) -> Any {
        let mut tables = crate::ffi::Fields::new();
        for group in &self.groups {
            tables = tables.with(&group.group.name, table(group));
        }
        tables.into_any()
    }

    /// Destroy the communicators collectively after every graph and call
    /// using them retired; an aborted close retains them until exit.
    pub fn close(&mut self, aborted: bool) -> Result<()> {
        let mut failure = None;
        for group in self.groups.drain(..) {
            let mut comm = group.comm.lock().unwrap_or_else(PoisonError::into_inner);
            if let Err(error) = comm.close(aborted) {
                failure.get_or_insert(error);
            }
        }
        failure.map_or(Ok(()), Err)
    }
}

impl Drop for StreamCollectives {
    fn drop(&mut self) {
        let _ = self.close(true);
    }
}

/// A device tensor argument: address, elements and NCCL type.
fn operand(value: &AnyView<'_>) -> Result<(u64, usize, DataType)> {
    let tensor = tvm_ffi::Tensor::try_from(*value)
        .map_err(|error| Error::numerical("collective", &error))?;
    if !tensor.is_contiguous() {
        return Err(Error::Invalid("collective tensors are contiguous".into()));
    }
    let dtype = tensor.dtype();
    let kind = match (dtype.code, dtype.bits) {
        (0, 8) => DataType::Int8,
        (0, 32) => DataType::Int32,
        (0, 64) => DataType::Int64,
        (1, 8) | (6, 8) => DataType::Uint8,
        (1, 32) => DataType::Uint32,
        (1, 64) => DataType::Uint64,
        (2, 16) => DataType::Float16,
        (2, 32) => DataType::Float32,
        (2, 64) => DataType::Float64,
        (4, 16) => DataType::Bfloat16,
        (10, 8) => DataType::Float8E4M3,
        (11, 8) => DataType::Float8E5M2,
        _ => {
            return Err(Error::Unsupported(format!(
                "collectives do not carry dtype code {} of {} bits",
                dtype.code, dtype.bits
            )));
        }
    };
    Ok((tensor.data_ptr() as u64, tensor.numel(), kind))
}

fn bytes_of(kind: DataType) -> usize {
    match kind {
        DataType::Int8 | DataType::Uint8 | DataType::Float8E4M3 | DataType::Float8E5M2 => 1,
        DataType::Float16 | DataType::Bfloat16 => 2,
        DataType::Int32 | DataType::Uint32 | DataType::Float32 => 4,
        DataType::Int64 | DataType::Uint64 | DataType::Float64 => 8,
    }
}

fn comm_size(comm: &StreamCommunicator<()>) -> Result<usize> {
    comm.size()
}

fn comm_rank(comm: &StreamCommunicator<()>) -> Result<i32> {
    comm.rank()
}

fn integer(value: &AnyView<'_>) -> Result<i64> {
    i64::try_from(*value).map_err(|error| Error::numerical("collective", &error))
}

fn integers(value: &AnyView<'_>) -> Result<Vec<i64>> {
    crate::ffi::integers(Any::from(*value)).map_err(|error| Error::numerical("collective", &error))
}

/// Started asynchronous collectives awaiting their consumer: the last token
/// issued and each pending token's completion event.
type Started = (i64, std::collections::HashMap<i64, Arc<Event>>);

/// The TVM-FFI functions of one group's communicator. Roots and peers are
/// process-world ranks; `all_to_all` splits count leading-axis rows.
fn table(group: &GroupComm) -> Any {
    type Call = dyn Fn(&mut StreamCommunicator<()>, &[AnyView<'_>]) -> Result<Any>;
    let mut ranks: Vec<usize> = group.group.ranks.clone();
    ranks.sort_unstable();
    let ranks = Arc::new(ranks);
    let tokens: Arc<std::sync::Mutex<Started>> = Arc::default();
    let function = |call: Box<Call>| {
        let comm = Arc::clone(&group.comm);
        tvm_ffi::Function::from_packed(move |args: &[AnyView<'_>]| {
            let mut comm = comm.lock().unwrap_or_else(PoisonError::into_inner);
            call(&mut comm, args).map_err(|error| {
                tvm_ffi::Error::new(tvm_ffi::RUNTIME_ERROR, &error.to_string(), "")
            })
        })
    };
    let peer = {
        let ranks = Arc::clone(&ranks);
        move |value: &AnyView<'_>| -> Result<i32> {
            let rank = integer(value)? as usize;
            ranks
                .iter()
                .position(|&member| member == rank)
                .map(|index| index as i32)
                .ok_or_else(|| Error::Invalid(format!("rank {rank} is outside the group")))
        }
    };
    let started = |asynchronous_all_to_all: bool| {
        let tokens = Arc::clone(&tokens);
        Box::new(
            move |comm: &mut StreamCommunicator<()>, args: &[AnyView<'_>]| -> Result<Any> {
                let [output, value] = args else {
                    return Err(Error::Invalid("takes an output and a value".into()));
                };
                let (output, _, _) = operand(output)?;
                let (input, count, kind) = operand(value)?;
                // SAFETY: both tensors are live device views the caller
                // keeps until it waits on the returned token.
                let event = unsafe {
                    if asynchronous_all_to_all {
                        comm.all_to_all(output, input, count / comm_size(comm)?, kind, true)
                    } else {
                        comm.all_gather(output, input, count, kind, true)
                    }
                }?
                .ok_or_else(|| {
                    Error::Invariant("an asynchronous collective has an event".into())
                })?;
                let mut tokens = tokens.lock().unwrap_or_else(PoisonError::into_inner);
                tokens.0 += 1;
                let token = tokens.0;
                tokens.1.insert(token, event);
                Ok(Any::from(token))
            },
        ) as Box<Call>
    };
    let wait = {
        let tokens = Arc::clone(&tokens);
        Box::new(
            move |comm: &mut StreamCommunicator<()>, args: &[AnyView<'_>]| -> Result<Any> {
                let [token, stream] = args else {
                    return Err(Error::Invalid("wait takes a token and a stream".into()));
                };
                let event = tokens
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .1
                    .remove(&integer(token)?)
                    .ok_or_else(|| Error::Invalid("unknown collective token".into()))?;
                // SAFETY: the binding names the stream it computes on, which
                // the rank's lanes own until the collectives close.
                let consumer =
                    unsafe { crate::stream::external(comm.device(), integer(stream)? as usize) }?;
                comm.join(&event, &consumer)?;
                Ok(Any::default())
            },
        ) as Box<Call>
    };
    let gather_peer = peer.clone();
    let broadcast_peer = peer.clone();
    let send_peer = peer.clone();
    let recv_peer = peer.clone();
    let pair_peer = peer;
    crate::ffi::Fields::new()
        .with(
            "all_reduce",
            function(Box::new(|comm, args| {
                let [value, op] = args else {
                    return Err(Error::Invalid("all_reduce takes a value and an op".into()));
                };
                let (data, count, kind) = operand(value)?;
                let op = match integer(op)? {
                    0 => ReduceOp::Sum,
                    2 => ReduceOp::Max,
                    3 => ReduceOp::Min,
                    other => return Err(Error::Invalid(format!("unknown reduction {other}"))),
                };
                // SAFETY: a live device tensor of `count` elements.
                unsafe { comm.all_reduce(data, count, kind, op) }?;
                Ok(Any::default())
            })),
        )
        .with(
            "all_gather",
            function(Box::new(|comm, args| {
                let [output, value] = args else {
                    return Err(Error::Invalid(
                        "all_gather takes an output and a value".into(),
                    ));
                };
                let (output, _, _) = operand(output)?;
                let (input, count, kind) = operand(value)?;
                // SAFETY: live device tensors sized for the group.
                unsafe { comm.all_gather(output, input, count, kind, false) }?;
                Ok(Any::default())
            })),
        )
        .with("start_all_gather", function(started(false)))
        .with("start_all_to_all", function(started(true)))
        .with("wait", function(wait))
        .with(
            "all_to_all",
            function(Box::new(|comm, args| {
                let [output, value, output_splits, input_splits] = args else {
                    return Err(Error::Invalid(
                        "all_to_all takes an output, a value and two split lists".into(),
                    ));
                };
                let (output, output_count, kind) = operand(output)?;
                let (input, input_count, _) = operand(value)?;
                let (output_splits, input_splits) =
                    (integers(output_splits)?, integers(input_splits)?);
                let size = comm_size(comm)?;
                let equal = output_splits.is_empty()
                    || (output_count == input_count
                        && output_splits
                            .iter()
                            .chain(&input_splits)
                            .all(|&split| split == output_splits[0]));
                if equal {
                    // SAFETY: live device tensors of `size` equal blocks.
                    unsafe { comm.all_to_all(output, input, input_count / size, kind, false) }?;
                    return Ok(Any::default());
                }
                let rows_out: i64 = output_splits.iter().sum();
                let rows_in: i64 = input_splits.iter().sum();
                let (row_out, row_in) = (
                    output_count / rows_out.max(1) as usize,
                    input_count / rows_in.max(1) as usize,
                );
                let element = bytes_of(kind);
                comm.grouped(|comm| {
                    let (mut sent, mut received) = (0_usize, 0_usize);
                    for peer in 0..size {
                        let rows_sent = input_splits[peer] as usize * row_in;
                        let rows_received = output_splits[peer] as usize * row_out;
                        // SAFETY: disjoint row ranges of live tensors.
                        unsafe {
                            if rows_sent > 0 {
                                comm.send(
                                    input + (sent * element) as u64,
                                    rows_sent,
                                    kind,
                                    peer as i32,
                                )?;
                            }
                            if rows_received > 0 {
                                comm.recv(
                                    output + (received * element) as u64,
                                    rows_received,
                                    kind,
                                    peer as i32,
                                )?;
                            }
                        }
                        sent += rows_sent;
                        received += rows_received;
                    }
                    Ok(())
                })?;
                Ok(Any::default())
            })),
        )
        .with(
            "gather",
            function(Box::new(move |comm, args| {
                let [outputs, value, root] = args else {
                    return Err(Error::Invalid(
                        "gather takes outputs, a value and a root".into(),
                    ));
                };
                let root = gather_peer(root)?;
                let (input, count, kind) = operand(value)?;
                let outputs: tvm_ffi::Array<Any> = tvm_ffi::Array::try_from(Any::from(*outputs))
                    .map_err(|error| Error::numerical("collective", &error))?;
                let outputs = outputs
                    .iter()
                    .map(|output| operand(&AnyView::from(&output)).map(|(address, ..)| address))
                    .collect::<Result<Vec<_>>>()?;
                let rank = comm_rank(comm)?;
                comm.grouped(|comm| {
                    // SAFETY: live device tensors of `count` elements each.
                    unsafe {
                        if rank == root {
                            for (peer, &output) in outputs.iter().enumerate() {
                                if peer as i32 == root {
                                    comm.send(input, count, kind, root)?;
                                }
                                comm.recv(output, count, kind, peer as i32)?;
                            }
                        } else {
                            comm.send(input, count, kind, root)?;
                        }
                    }
                    Ok(())
                })?;
                Ok(Any::default())
            })),
        )
        .with(
            "broadcast",
            function(Box::new(move |comm, args| {
                let [value, root] = args else {
                    return Err(Error::Invalid("broadcast takes a value and a root".into()));
                };
                let (data, count, kind) = operand(value)?;
                // SAFETY: a live device tensor of `count` elements.
                unsafe { comm.broadcast(data, count, kind, broadcast_peer(root)?) }?;
                Ok(Any::default())
            })),
        )
        .with(
            "reduce_scatter",
            function(Box::new(|comm, args| {
                let [output, value] = args else {
                    return Err(Error::Invalid(
                        "reduce_scatter takes an output and a value".into(),
                    ));
                };
                let (output, count, kind) = operand(output)?;
                let (input, _, _) = operand(value)?;
                // SAFETY: live device tensors sized for the group.
                unsafe { comm.reduce_scatter(output, input, count, kind) }?;
                Ok(Any::default())
            })),
        )
        .with(
            "send",
            function(Box::new(move |comm, args| {
                let [value, peer] = args else {
                    return Err(Error::Invalid("send takes a value and a peer".into()));
                };
                let (data, count, kind) = operand(value)?;
                // SAFETY: a live device tensor of `count` elements.
                unsafe { comm.send(data, count, kind, send_peer(peer)?) }?;
                Ok(Any::default())
            })),
        )
        .with(
            "recv",
            function(Box::new(move |comm, args| {
                let [value, peer] = args else {
                    return Err(Error::Invalid("recv takes a value and a peer".into()));
                };
                let (data, count, kind) = operand(value)?;
                // SAFETY: a live device tensor of `count` elements.
                unsafe { comm.recv(data, count, kind, recv_peer(peer)?) }?;
                Ok(Any::default())
            })),
        )
        .with(
            "send_recv",
            function(Box::new(move |comm, args| {
                let [output, value, destination, source] = args else {
                    return Err(Error::Invalid(
                        "send_recv takes an output, a value, a destination and a source".into(),
                    ));
                };
                let (output, output_count, output_kind) = operand(output)?;
                let (input, input_count, input_kind) = operand(value)?;
                let (destination, source) = (pair_peer(destination)?, pair_peer(source)?);
                comm.grouped(|comm| {
                    // SAFETY: live device tensors, exchanged as bytes.
                    unsafe {
                        comm.send(
                            input,
                            input_count * bytes_of(input_kind),
                            DataType::Uint8,
                            destination,
                        )?;
                        comm.recv(
                            output,
                            output_count * bytes_of(output_kind),
                            DataType::Uint8,
                            source,
                        )?;
                    }
                    Ok(())
                })?;
                Ok(Any::default())
            })),
        )
        .into_any()
}

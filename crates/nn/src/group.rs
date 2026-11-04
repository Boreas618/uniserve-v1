//! The communicator a partitioned block runs its collectives through.
//!
//! The design's [`Forward`](crate::Forward) carries `Option<&Communicator>` (specs/worker.md 4.3):
//! a non-owning borrowing of the NCCL communicator the executor created for this rank. The CUDA
//! boundary's collectives take `&mut Communicator` because NCCL calls into one communicator must
//! not interleave, while a forward holds its resources immutably. [`Group`] is the box the executor
//! puts its communicator in: it owns the communicator, serializes access to it on the one executor
//! thread, and offers the collectives a layer needs through `&self`.
//!
//! One executor thread issues every job of its device (specs/worker.md 2.4), so the interior
//! mutability here is not contention: it is the type-level statement that a forward borrows the
//! group and does not own it.

use std::cell::RefCell;

use ops::platform::{DeviceRange, PlatformDevicePtr};
use ops::{CollectiveType, Communicator, ReduceOp, Stream};

use crate::error::{Error, Result};
use crate::tensor::TensorView;

/// The bytes `view` addresses, as the collective's range type wants them.
fn range(view: TensorView, bytes: usize) -> DeviceRange {
    DeviceRange::new(PlatformDevicePtr::from_address(view.ptr.address()), bytes)
}

/// A rank's NCCL communicator, borrowed by forwards.
pub struct Group {
    communicator: RefCell<Communicator>,
    ranks: u64,
}

impl Group {
    /// The group of `communicator`, whose member count is `ranks`.
    pub fn new(communicator: Communicator, ranks: u64) -> Self {
        Self {
            communicator: RefCell::new(communicator),
            ranks,
        }
    }

    /// How many ranks the group has.
    pub const fn ranks(&self) -> u64 {
        self.ranks
    }

    /// This rank's index, as NCCL reports it.
    pub fn rank(&self) -> i32 {
        self.communicator.borrow().rank()
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
        self.communicator
            .borrow_mut()
            .all_reduce(
                stream,
                range(send, bytes),
                range(recv, bytes),
                count,
                dtype,
                op,
            )
            .map_err(Error::Platform)
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
        self.communicator
            .borrow_mut()
            .all_gather(
                stream,
                range(send, bytes * count),
                range(recv, bytes * expected),
                count,
                dtype,
            )
            .map_err(Error::Platform)
    }

    /// Exchange equal blocks with every rank: block `j` goes to rank `j`, rank `j`'s comes back.
    pub fn all_to_all(&self, send: TensorView, recv: TensorView, stream: &Stream) -> Result<()> {
        let (count, dtype) = self.collective(send, recv, "all_to_all")?;
        let bytes = dtype.bytes() * count * usize::try_from(self.ranks).unwrap_or(usize::MAX);
        self.communicator
            .borrow_mut()
            .all_to_all(stream, range(send, bytes), range(recv, bytes), count, dtype)
            .map_err(Error::Platform)
    }

    /// Run `call` with the underlying communicator, for an operator whose collective is not one of
    /// the three above (`ops::exchange`, an expert transport).
    pub fn with<R>(&self, call: impl FnOnce(&mut Communicator) -> Result<R>) -> Result<R> {
        call(&mut self.communicator.borrow_mut())
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

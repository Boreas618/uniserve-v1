//! Stream-ordered NCCL communicators.

mod ffi;

use std::ffi::c_void;

use ffi::{Handle, RawUniqueId, api};

use crate::{DeviceId, Error, PrimaryContext, Result, Stream};

/// `ncclCommSuspend`'s flag that releases the communicator's dynamic device
/// storage.
const NCCL_SUSPEND_MEM: i32 = 0x01;

/// The 128-byte rendezvous token every rank of a communicator initializes
/// with. Rank 0 creates it and the worker distributes it to its peers.
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
        let mut id = RawUniqueId { bytes: [0; 128] };
        // SAFETY: NCCL writes the 128-byte id.
        api.check(unsafe { (api.get_unique_id)(&mut id) }, "ncclGetUniqueId")?;
        Ok(Self(id.bytes))
    }
}

/// `ncclDataType_t`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum DataType {
    Int8 = 0,
    Uint8 = 1,
    Int32 = 2,
    Uint32 = 3,
    Int64 = 4,
    Uint64 = 5,
    Float16 = 6,
    Float32 = 7,
    Float64 = 8,
    Bfloat16 = 9,
    Float8E4M3 = 10,
    Float8E5M2 = 11,
}

/// `ncclRedOp_t` (built-in reductions).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum ReduceOp {
    Sum = 0,
    Prod = 1,
    Max = 2,
    Min = 3,
    Avg = 4,
}

/// One rank's NCCL communicator.
///
/// Every operation is enqueued on the stream passed to it, which belongs to
/// the communicator's device. Collective calls must be issued in the same
/// order on every rank; group several with [`group`].
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
// forbids concurrent calls on one communicator, which `&mut self` on every
// operation enforces. It is therefore Send but not Sync.
unsafe impl Send for Communicator {}

impl Communicator {
    /// Joins communicator `id` as `rank` of `size` on `device`. Every rank
    /// calls this once with the same id; the call blocks until all joined.
    ///
    /// The communicator takes NCCL's default configuration, which `NCCL_*`
    /// environment variables tune. Its collectives run as NCCL kernels on
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
        // SAFETY: NCCL writes one communicator handle.
        let status =
            unsafe { (api.comm_init_rank)(&mut raw, size, RawUniqueId { bytes: id.0 }, rank) };
        if let Err(error) = api.check(status, "ncclCommInitRank") {
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

    /// Reduces `count` elements from every rank into every rank's `recv`
    /// (in place when `send == recv`).
    ///
    /// # Safety
    /// Both buffers hold `count` elements of `dtype` on the communicator's
    /// device and stay valid until the stream (or graph replay) completes.
    pub unsafe fn all_reduce(
        &mut self,
        send: u64,
        recv: u64,
        count: usize,
        dtype: DataType,
        op: ReduceOp,
        stream: &Stream,
    ) -> Result<()> {
        let api = api()?;
        // SAFETY: forwarded from the caller.
        let status = unsafe {
            (api.all_reduce)(
                pointer(send),
                pointer(recv),
                count,
                dtype as i32,
                op as i32,
                self.raw,
                stream.handle(),
            )
        };
        api.check(status, "ncclAllReduce")
    }

    /// Reduces `size * count` input elements and leaves block `rank` (`count`
    /// elements) in `recv`.
    ///
    /// # Safety
    /// `send` holds `size * count` and `recv` holds `count` elements; both
    /// stay valid until the stream (or graph replay) completes.
    pub unsafe fn reduce_scatter(
        &mut self,
        send: u64,
        recv: u64,
        count: usize,
        dtype: DataType,
        op: ReduceOp,
        stream: &Stream,
    ) -> Result<()> {
        let api = api()?;
        // SAFETY: forwarded from the caller.
        let status = unsafe {
            (api.reduce_scatter)(
                pointer(send),
                pointer(recv),
                count,
                dtype as i32,
                op as i32,
                self.raw,
                stream.handle(),
            )
        };
        api.check(status, "ncclReduceScatter")
    }

    /// Gathers `count` elements from every rank into `recv`, rank-major.
    ///
    /// # Safety
    /// `send` holds `count` and `recv` holds `size * count` elements; both
    /// stay valid until the stream (or graph replay) completes.
    pub unsafe fn all_gather(
        &mut self,
        send: u64,
        recv: u64,
        count: usize,
        dtype: DataType,
        stream: &Stream,
    ) -> Result<()> {
        let api = api()?;
        // SAFETY: forwarded from the caller.
        let status = unsafe {
            (api.all_gather)(
                pointer(send),
                pointer(recv),
                count,
                dtype as i32,
                self.raw,
                stream.handle(),
            )
        };
        api.check(status, "ncclAllGather")
    }

    /// Sends block `j` (`count` elements) of `send` to rank `j` and receives
    /// rank `j`'s block into block `j` of `recv`.
    ///
    /// # Safety
    /// Both buffers hold `size * count` elements and stay valid until the
    /// stream (or graph replay) completes.
    pub unsafe fn all_to_all(
        &mut self,
        send: u64,
        recv: u64,
        count: usize,
        dtype: DataType,
        stream: &Stream,
    ) -> Result<()> {
        let api = api()?;
        let all_to_all = api.all_to_all.clone()?;
        // SAFETY: forwarded from the caller.
        let status = unsafe {
            all_to_all(
                pointer(send),
                pointer(recv),
                count,
                dtype as i32,
                self.raw,
                stream.handle(),
            )
        };
        api.check(status, "ncclAlltoAll")
    }

    /// Copies `count` elements of `root`'s `send` into every rank's `recv`.
    ///
    /// # Safety
    /// Both buffers hold `count` elements (`send` is read on `root` only) and
    /// stay valid until the stream (or graph replay) completes.
    pub unsafe fn broadcast(
        &mut self,
        send: u64,
        recv: u64,
        count: usize,
        dtype: DataType,
        root: i32,
        stream: &Stream,
    ) -> Result<()> {
        let api = api()?;
        // SAFETY: forwarded from the caller.
        let status = unsafe {
            (api.broadcast)(
                pointer(send),
                pointer(recv),
                count,
                dtype as i32,
                root,
                self.raw,
                stream.handle(),
            )
        };
        api.check(status, "ncclBroadcast")
    }

    /// Sends `count` elements to `peer`, which posts a matching [`Self::recv`].
    ///
    /// # Safety
    /// The buffer stays valid until the stream (or graph replay) completes.
    pub unsafe fn send(
        &mut self,
        data: u64,
        count: usize,
        dtype: DataType,
        peer: i32,
        stream: &Stream,
    ) -> Result<()> {
        let api = api()?;
        // SAFETY: forwarded from the caller.
        let status = unsafe {
            (api.send)(
                pointer(data),
                count,
                dtype as i32,
                peer,
                self.raw,
                stream.handle(),
            )
        };
        api.check(status, "ncclSend")
    }

    /// Receives `count` elements from `peer`.
    ///
    /// # Safety
    /// The buffer stays valid until the stream (or graph replay) completes.
    pub unsafe fn recv(
        &mut self,
        data: u64,
        count: usize,
        dtype: DataType,
        peer: i32,
        stream: &Stream,
    ) -> Result<()> {
        let api = api()?;
        // SAFETY: forwarded from the caller.
        let status = unsafe {
            (api.recv)(
                pointer(data),
                count,
                dtype as i32,
                peer,
                self.raw,
                stream.handle(),
            )
        };
        api.check(status, "ncclRecv")
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

    /// Releases the communicator's dynamic device storage
    /// (`ncclCommSuspend(NCCL_SUSPEND_MEM)`) until [`Self::resume`]. No
    /// operation may run on it, nor a graph that captured one, in between.
    ///
    /// # Errors
    ///
    /// Fails when the loaded NCCL predates suspension (2.30) or refuses it.
    pub fn suspend(&mut self) -> Result<()> {
        let api = api()?;
        let suspend = api.comm_suspend.as_ref().map_err(Clone::clone)?;
        let _guard = self.context.enter()?;
        // SAFETY: the communicator is live and idle by the caller's contract.
        api.check(
            unsafe { suspend(self.raw, NCCL_SUSPEND_MEM) },
            "ncclCommSuspend",
        )
    }

    /// Allocates again what [`Self::suspend`] released; the communicator,
    /// and graphs that captured its operations, run afterwards.
    pub fn resume(&mut self) -> Result<()> {
        let api = api()?;
        let resume = api.comm_resume.as_ref().map_err(Clone::clone)?;
        let _guard = self.context.enter()?;
        // SAFETY: the communicator is live and suspended.
        api.check(unsafe { resume(self.raw) }, "ncclCommResume")
    }

    /// Tears down this rank's communicator without waiting for peers, aborting
    /// its pending operations.
    pub fn abort(self) -> Result<()> {
        let api = api()?;
        let _guard = self.context.enter()?;
        // SAFETY: the communicator is owned and aborted once.
        api.check(unsafe { (api.comm_abort)(self.raw) }, "ncclCommAbort")
    }
}

fn pointer(address: u64) -> Handle {
    address as usize as *mut c_void
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

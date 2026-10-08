//! Stream-ordered NCCL communicators and NCCL-allocated buffers.

mod ffi;

use std::ffi::c_void;

use ffi::{Handle, RawUniqueId, api};

use crate::driver::in_context;
use crate::{DeviceId, Error, PrimaryContext, Result, Stream, reclaim};

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

/// How NCCL assigns CTAs (`ncclConfig_t::CTAPolicy`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CtaPolicy {
    /// NCCL's kernel-based algorithms. Communicators of SM-partitioned
    /// (green-context) streams use this: green contexts cannot batch-copy
    /// mapped peer windows on the deployed driver.
    #[default]
    Default,
    /// Copy-engine algorithms for registered symmetric windows, leaving SMs
    /// to computation. Used by communicators of full-device streams.
    Zero,
}

impl CtaPolicy {
    fn raw(self) -> i32 {
        match self {
            Self::Default => 0x00,
            Self::Zero => 0x02,
        }
    }
}

/// One rank's NCCL communicator.
///
/// Every operation is enqueued on the stream passed to it, which belongs to
/// the communicator's device. Collective calls must be issued in the same
/// order on every rank; group several with [`group`].
///
/// Release is explicit because it is collective: [`Communicator::destroy`]
/// deregisters windows and destroys the communicator once every rank's work
/// retired, and [`Communicator::abort`] tears it down locally after a
/// failure. Dropping a communicator that was neither destroyed nor aborted
/// leaks it deliberately: an implicit destroy could wait forever for peers
/// that never arrive, and an abort could free resources that captured graphs
/// still use.
pub struct Communicator {
    raw: Handle,
    context: PrimaryContext,
    rank: i32,
    size: i32,
    windows: Vec<Window>,
}

struct Window {
    address: u64,
    bytes: usize,
    raw: Handle,
}

// SAFETY: a communicator may be used from any one thread at a time; NCCL
// forbids concurrent calls on one communicator, which `&mut self` on every
// operation enforces. It is therefore Send but not Sync.
unsafe impl Send for Communicator {}

impl Communicator {
    /// Joins communicator `id` as `rank` of `size` on `device`. Every rank
    /// calls this once with the same id; the call blocks until all joined.
    pub fn init(
        device: DeviceId,
        size: i32,
        rank: i32,
        id: &UniqueId,
        policy: CtaPolicy,
    ) -> Result<Self> {
        if size <= 0 || !(0..size).contains(&rank) {
            return Err(Error::invalid(format!(
                "rank {rank} is outside a communicator of {size} ranks"
            )));
        }
        let api = api()?;
        let context = PrimaryContext::retain(device)?;
        let _guard = context.enter()?;

        let mut raw = std::ptr::null_mut();
        let mut config = ffi::Config::new(policy.raw());
        // SAFETY: NCCL writes one communicator handle and reads the config.
        let status = unsafe {
            (api.comm_init_rank_config)(
                &mut raw,
                size,
                RawUniqueId { bytes: id.0 },
                rank,
                &mut config,
            )
        };
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
            windows: Vec::new(),
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

    /// Registers `[address, address + bytes)` as a symmetric window
    /// (`NCCL_WIN_COLL_SYMMETRIC`). Registering the same range twice is a
    /// no-op. Windows are deregistered by [`Self::destroy`].
    ///
    /// # Safety
    /// The range is VMM-backed (for example a [`Buffer`]) and stays allocated
    /// until the communicator is destroyed or aborted. Every rank registers
    /// matching windows in the same order, outside graph capture.
    pub unsafe fn register_window(&mut self, address: u64, bytes: usize) -> Result<()> {
        const NCCL_WIN_COLL_SYMMETRIC: i32 = 0x01;

        if self
            .windows
            .iter()
            .any(|window| window.address == address && window.bytes == bytes)
        {
            return Ok(());
        }
        let api = api()?;
        let register = api.window_register.clone()?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: the caller guarantees the range; NCCL writes one window.
        let status = unsafe {
            register(
                self.raw,
                pointer(address),
                bytes,
                &mut raw,
                NCCL_WIN_COLL_SYMMETRIC,
            )
        };
        api.check(status, "ncclCommWindowRegister")?;

        self.windows.push(Window {
            address,
            bytes,
            raw,
        });
        Ok(())
    }

    /// Deregisters every window and destroys the communicator. Collective:
    /// every rank calls it after all work using the communicator, including
    /// graphs that captured it, has retired. On failure the communicator's
    /// remaining resources are leaked.
    pub fn destroy(self) -> Result<()> {
        let api = api()?;
        let _guard = self.context.enter()?;

        for window in &self.windows {
            let deregister = api.window_deregister.clone()?;
            // SAFETY: the window was registered on this communicator.
            api.check(
                unsafe { deregister(self.raw, window.raw) },
                "ncclCommWindowDeregister",
            )?;
        }
        // SAFETY: the communicator is owned and destroyed once.
        api.check(unsafe { (api.comm_destroy)(self.raw) }, "ncclCommDestroy")
    }

    /// Tears down this rank's communicator without waiting for peers, aborting
    /// its pending operations. Windows need no deregistration.
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

/// Device memory from `ncclMemAlloc`: VMM-backed and suitable for symmetric
/// window registration.
///
/// Dropping queues `ncclMemFree` to the [`crate::reclaim`]er; the owner
/// guarantees that no device work and no live window still use the buffer.
pub struct Buffer {
    context: PrimaryContext,
    address: u64,
    bytes: usize,
}

impl Buffer {
    pub fn new(device: DeviceId, bytes: usize) -> Result<Self> {
        if bytes == 0 {
            return Err(Error::invalid("an allocation needs a positive length"));
        }
        let api = api()?;
        let allocate = api.mem_alloc.clone()?;
        // Fail before allocating if the release entry point is missing.
        api.mem_free.clone()?;
        let context = PrimaryContext::retain(device)?;
        let _guard = context.enter()?;

        let mut raw = std::ptr::null_mut();
        // SAFETY: NCCL writes the base of a new allocation on the current device.
        api.check(unsafe { allocate(&mut raw, bytes) }, "ncclMemAlloc")?;

        Ok(Self {
            context,
            address: raw as u64,
            bytes,
        })
    }

    pub fn device(&self) -> DeviceId {
        self.context.device()
    }

    pub fn address(&self) -> u64 {
        self.address
    }

    pub fn len(&self) -> usize {
        self.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let (context, address) = (self.context.clone(), self.address);
        reclaim::defer(move || {
            let api = api()?;
            let free = api.mem_free.clone()?;
            in_context(context.raw(), |_| {
                // SAFETY: the allocation is owned and freed once.
                api.check(unsafe { free(pointer(address)) }, "ncclMemFree")
            })
        });
    }
}
